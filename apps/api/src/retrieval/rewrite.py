"""Query rewriting: HyDE, multi-query expansion and decomposition.

The gap these close is that a user's question and the passage that answers it are
written in different registers. "Why is my build slow?" and "Incremental
compilation is disabled when the cache directory is not writable" share almost no
vocabulary, so neither BM25 nor a bi-encoder reliably connects them.

Three techniques, each for a different failure:

* **HyDE** writes a hypothetical *answer* and embeds that instead of the
  question, so the query vector lands in answer-space where the documents live.
  It hallucinates freely and that is fine — the text is never shown to anyone,
  only embedded.
* **Multi-query** generates paraphrases to cover vocabulary the user did not
  happen to use. Fusing several ranked lists beats guessing one right phrasing.
* **Decomposition** splits a multi-hop question into independently answerable
  sub-questions, because no single retrieval can satisfy "how does X compare to
  Y" when X and Y live in different documents.

All three fail soft: if the rewrite call fails, retrieval proceeds with the
original query rather than failing the turn.

Example:
    >>> QueryRewriter.HYDE_PROMPT_NAME
    'hyde'
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.core.logging import get_logger
from src.services.llm.router import LLMRouter
from src.services.llm.types import CompletionRequest, Message

log = get_logger(__name__)

#: Cap on generated variants. Beyond about five, fusion gains flatten while cost
#: and latency keep climbing.
MAX_VARIANTS = 5

_HYDE_SYSTEM = (
    "Write a short, factual passage that would directly answer the user's question, "
    "as if excerpted from an internal document. Two to four sentences. Use the "
    "vocabulary such a document would use. Do not hedge, do not mention that this "
    "is hypothetical, and do not address the user."
)

_MULTI_QUERY_SYSTEM = (
    "Rewrite the user's question into {n} alternative search queries that use "
    "different vocabulary but seek the same information. Vary between formal and "
    "informal phrasing and between specific and general wording. Output one query "
    "per line, with no numbering, quotes or commentary."
)

_DECOMPOSE_SYSTEM = (
    "Break the user's question into the minimum set of standalone sub-questions "
    "needed to answer it. Each sub-question must be answerable on its own, without "
    "reference to the others. If the question is already atomic, output it "
    "unchanged. Output one sub-question per line, with no numbering or commentary."
)


@dataclass(slots=True)
class RewriteResult:
    """The product of a rewriting pass."""

    original: str
    #: Extra query strings to search alongside the original.
    expansions: tuple[str, ...] = ()
    #: Sub-questions for multi-hop queries; empty when the query is atomic.
    sub_questions: tuple[str, ...] = ()
    #: The generated hypothetical document, kept for tracing but never displayed.
    hypothetical_document: str | None = None
    techniques: tuple[str, ...] = ()
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def all_queries(self) -> tuple[str, ...]:
        """Original plus expansions, de-duplicated and order-preserving.

        Example:
            >>> RewriteResult(original="a", expansions=("b", "a")).all_queries
            ('a', 'b')
        """
        seen = {self.original: None}
        for expansion in self.expansions:
            seen.setdefault(expansion, None)
        return tuple(seen)

    @property
    def is_multi_hop(self) -> bool:
        """True when decomposition produced more than one sub-question."""
        return len(self.sub_questions) > 1


class QueryRewriter:
    """Generates query variants using a cheap model."""

    HYDE_PROMPT_NAME = "hyde"
    MULTI_QUERY_PROMPT_NAME = "multi_query"
    DECOMPOSE_PROMPT_NAME = "decompose"

    def __init__(self, *, router: LLMRouter, model: str | None = None) -> None:
        """Create a rewriter.

        Args:
            router: Router for the generation calls.
            model: Model to use. Rewriting is a cheap-model job; using a frontier
                model here roughly doubles the cost of a turn for no measurable
                retrieval gain.
        """
        self._router = router
        self._model = model

    async def rewrite(
        self,
        query: str,
        *,
        use_hyde: bool = False,
        use_multi_query: bool = False,
        decompose: bool = False,
        variants: int = 3,
    ) -> RewriteResult:
        """Apply the enabled rewriting techniques to a query.

        Techniques compose: HyDE and multi-query both contribute expansions, and
        decomposition additionally reports sub-questions for the agent to iterate
        over.
        """
        result = RewriteResult(original=query)
        expansions: list[str] = []
        techniques: list[str] = []

        if use_hyde:
            document = await self._safe_generate(
                _HYDE_SYSTEM, query, max_tokens=220, label="hyde", result=result
            )
            if document:
                result.hypothetical_document = document
                expansions.append(document)
                techniques.append("hyde")

        if use_multi_query:
            n = max(1, min(variants, MAX_VARIANTS))
            raw = await self._safe_generate(
                _MULTI_QUERY_SYSTEM.format(n=n),
                query,
                max_tokens=200,
                label="multi_query",
                result=result,
            )
            if raw:
                generated = _parse_lines(raw, limit=n, exclude=query)
                expansions.extend(generated)
                if generated:
                    techniques.append("multi_query")

        if decompose:
            raw = await self._safe_generate(
                _DECOMPOSE_SYSTEM, query, max_tokens=220, label="decompose", result=result
            )
            if raw:
                parts = _parse_lines(raw, limit=MAX_VARIANTS, exclude=None)
                result.sub_questions = tuple(parts)
                if len(parts) > 1:
                    techniques.append("decompose")

        result.expansions = tuple(dict.fromkeys(expansions))
        result.techniques = tuple(techniques)
        return result

    async def _safe_generate(
        self,
        system: str,
        query: str,
        *,
        max_tokens: int,
        label: str,
        result: RewriteResult,
    ) -> str | None:
        """Run one rewrite call, recording failure instead of raising.

        A rewrite is an optimisation. Losing it costs recall; failing the turn
        costs the user their answer.
        """
        request = CompletionRequest(
            messages=(Message.system(system), Message.user(query)),
            model=self._model or "",
            max_tokens=max_tokens,
            temperature=0.3,
            node=f"query_rewriter.{label}",
        )
        try:
            completion = await self._router.complete(
                request.model_copy(update={"model": self._model or request.model}),
                allow_fallback=False,
            )
        except Exception as exc:  # noqa: BLE001 - rewriting is best-effort
            log.warning(
                "query rewrite failed; continuing without it", technique=label, reason=str(exc)
            )
            result.errors.append(f"{label}: {exc}")
            return None

        result.cost_usd += completion.cost_usd
        return completion.content.strip() or None


def _parse_lines(raw: str, *, limit: int, exclude: str | None) -> list[str]:
    """Parse a newline-separated model response into clean query strings.

    Models add numbering, bullets and quotes no matter how firmly the prompt says
    not to, so stripping them here is more reliable than prompting harder.

    Example:
        >>> raw = chr(10).join(["1. first query", "- second query"])
        >>> _parse_lines(raw, limit=5, exclude=None)
        ['first query', 'second query']
        >>> _parse_lines('"only one"', limit=5, exclude="only one")
        []
    """
    cleaned: list[str] = []
    normalised_exclude = (exclude or "").strip().lower()

    for line in raw.splitlines():
        candidate = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip().strip("\"'")
        if not candidate or len(candidate) < 3:
            continue
        if candidate.strip().lower() == normalised_exclude:
            continue
        if candidate not in cleaned:
            cleaned.append(candidate)
        if len(cleaned) >= limit:
            break
    return cleaned


def looks_time_sensitive(query: str) -> bool:
    """Heuristic for whether a query wants recent documents.

    Runs before any model call because it decides whether to *ask* for a recency
    boost, and paying for a classifier to decide whether to pay for a classifier
    is not a good trade.

    Example:
        >>> looks_time_sensitive("what is our latest pricing?")
        True
        >>> looks_time_sensitive("what does RAG stand for?")
        False
    """
    markers = (
        "latest", "current", "recent", "now", "today", "this year", "this quarter",
        "up to date", "up-to-date", "newest", "as of", "still", "changed", "new",
    )  # fmt: skip
    lowered = query.lower()
    return any(marker in lowered for marker in markers)


def looks_multi_hop(query: str) -> bool:
    """Heuristic for whether a query needs decomposition.

    Cheap pre-filter so the decomposition call is only made when it is plausibly
    useful; the router's classifier makes the real decision.

    Example:
        >>> looks_multi_hop("compare our EMEA and APAC revenue")
        True
        >>> looks_multi_hop("what is our EMEA revenue?")
        False
    """
    markers = (
        "compare", "versus", " vs ", "difference between", "both", "and also",
        "as well as", "which of", "rank", "trade-off", "tradeoff", "relative to",
    )  # fmt: skip
    lowered = f" {query.lower()} "
    return any(marker in lowered for marker in markers)
