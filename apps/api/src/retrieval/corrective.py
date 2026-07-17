"""Corrective RAG: judge the retrieval before trusting it.

Standard RAG has no opinion about whether what it retrieved is any good. It
passes the top-k to the generator regardless, and the generator — which has no
way to know the context is irrelevant — answers from it anyway. That is the
mechanism behind a large share of confident wrong answers.

CRAG (Yan et al., 2024) inserts an evaluator between retrieval and generation
which grades the retrieved set and picks one of three routes:

* **correct** — the context answers the question; generate from it.
* **ambiguous** — partially relevant; keep it *and* widen the search, because
  discarding partially-correct context loses real signal.
* **incorrect** — nothing relevant; fall back to web search rather than inviting
  the generator to invent an answer from unrelated passages.

The evaluator is a cheap model with a score threshold, not a frontier model:
grading relevance is much easier than answering, and putting an expensive call
in front of every turn would defeat the purpose.

Example:
    >>> verdict_from_score(0.9, thresholds=CragThresholds())
    <CragVerdict.CORRECT: 'correct'>
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.core.logging import get_logger
from src.retrieval.types import CragVerdict, RetrievedChunk
from src.services.llm.router import LLMRouter
from src.services.llm.types import CompletionRequest, Message

log = get_logger(__name__)

_SYSTEM = (
    "You grade whether a retrieved passage helps answer a question. Reply with a "
    "single number from 0 to 10 and nothing else. 0 means the passage is unrelated. "
    "10 means it directly and completely answers the question. Grade only what the "
    "passage contains; do not use outside knowledge."
)

_USER = """Question: {query}

Passage:
{passage}

Score (0-10):"""


class CragThresholds(BaseModel):
    """Score boundaries between the three CRAG routes.

    The defaults are deliberately asymmetric. ``incorrect`` sits low because
    triggering a web search on a merely-mediocre retrieval is slow and expensive;
    ``correct`` sits high because passing weak context to the generator is the
    exact failure this node exists to prevent.
    """

    model_config = ConfigDict(frozen=True)

    correct_at_or_above: float = Field(default=0.7, ge=0.0, le=1.0)
    incorrect_below: float = Field(default=0.3, ge=0.0, le=1.0)
    #: Fraction of graded chunks that must clear ``correct_at_or_above`` for the
    #: set as a whole to count as correct.
    min_relevant_fraction: float = Field(default=0.34, ge=0.0, le=1.0)


@dataclass(slots=True)
class CragAssessment:
    """The evaluator's verdict on one retrieval."""

    verdict: CragVerdict
    scores: tuple[float, ...]
    kept: tuple[RetrievedChunk, ...]
    cost_usd: float = 0.0
    graded_with: str = "llm"

    @property
    def top_score(self) -> float:
        """Best per-chunk relevance score, or 0.0 when nothing was graded."""
        return max(self.scores) if self.scores else 0.0

    @property
    def needs_web_search(self) -> bool:
        """Whether the agent should fall back to the web."""
        return self.verdict is CragVerdict.INCORRECT

    @property
    def needs_wider_retrieval(self) -> bool:
        """Whether the agent should widen k and retrieve again."""
        return self.verdict is CragVerdict.AMBIGUOUS


class WebSearchClient(Protocol):
    """Minimal web-search interface the fallback depends on."""

    async def search(self, query: str, *, max_results: int) -> list[RetrievedChunk]:
        """Return web results already shaped as retrieval hits."""
        ...


class RetrievalEvaluator:
    """Grades retrieved chunks and chooses the corrective route."""

    PROMPT_NAME = "crag_evaluator"

    def __init__(
        self,
        *,
        router: LLMRouter,
        model: str | None = None,
        thresholds: CragThresholds | None = None,
        max_graded: int = 8,
    ) -> None:
        """Create an evaluator.

        Args:
            router: Router for the grading calls.
            model: Grading model; defaults to the router's cheap model.
            thresholds: Route boundaries.
            max_graded: Chunks graded per turn. Grading is one call per chunk, so
                this is the node's cost ceiling.
        """
        self._router = router
        self._model = model
        self._thresholds = thresholds or CragThresholds()
        self._max_graded = max_graded

    async def evaluate(self, query: str, chunks: Sequence[RetrievedChunk]) -> CragAssessment:
        """Grade the retrieved set and return a verdict.

        An empty retrieval is ``incorrect`` without spending anything: there is
        nothing to grade, and the answer is obvious.
        """
        if not chunks:
            return CragAssessment(
                verdict=CragVerdict.INCORRECT, scores=(), kept=(), graded_with="empty"
            )

        import asyncio

        graded = list(chunks[: self._max_graded])
        results = await asyncio.gather(*(self._grade(query, chunk) for chunk in graded))
        scores = tuple(score for score, _ in results)
        cost = sum(spend for _, spend in results)

        # Keep partially-relevant context: discarding it on an ambiguous verdict
        # throws away the signal that made the verdict ambiguous in the first place.
        kept = tuple(
            chunk
            for chunk, score in zip(graded, scores, strict=True)
            if score >= self._thresholds.incorrect_below
        )
        verdict = self._verdict_for(scores)

        log.info(
            "crag assessment",
            verdict=verdict.value,
            graded=len(graded),
            kept=len(kept),
            top_score=round(max(scores) if scores else 0.0, 3),
        )
        return CragAssessment(verdict=verdict, scores=scores, kept=kept, cost_usd=cost)

    def _verdict_for(self, scores: Sequence[float]) -> CragVerdict:
        """Aggregate per-chunk scores into one verdict."""
        if not scores:
            return CragVerdict.INCORRECT

        relevant = sum(1 for s in scores if s >= self._thresholds.correct_at_or_above)
        fraction = relevant / len(scores)

        if relevant and fraction >= self._thresholds.min_relevant_fraction:
            return CragVerdict.CORRECT
        if max(scores) < self._thresholds.incorrect_below:
            return CragVerdict.INCORRECT
        return CragVerdict.AMBIGUOUS

    async def _grade(self, query: str, chunk: RetrievedChunk) -> tuple[float, float]:
        """Grade one chunk, returning ``(score, cost_usd)``.

        A grading failure returns a neutral score rather than zero: treating an
        outage as "irrelevant" would silently route every turn to web search.
        """
        request = CompletionRequest(
            messages=(
                Message.system(_SYSTEM),
                Message.user(_USER.format(query=query, passage=chunk.content[:4000])),
            ),
            model=self._model or "",
            max_tokens=8,
            temperature=0.0,
            node="retrieval_evaluator",
        )
        try:
            completion = await self._router.complete(
                request.model_copy(update={"model": self._model or request.model}),
                allow_fallback=False,
            )
        except Exception as exc:  # noqa: BLE001 - grading is best-effort
            log.warning("crag grading failed; scoring neutral", reason=str(exc))
            return 0.5, 0.0

        return parse_score(completion.content), completion.cost_usd


def parse_score(raw: str) -> float:
    """Parse a 0-10 grade into a 0-1 score, tolerating chatty models.

    Returns a neutral 0.5 when nothing numeric can be found, for the same reason
    a failed call does: an unparseable response is not evidence of irrelevance.

    Example:
        >>> parse_score("8")
        0.8
        >>> parse_score("Score: 10/10")
        1.0
        >>> parse_score("I cannot grade this")
        0.5
        >>> parse_score("42")
        1.0
    """
    match = re.search(r"\d+(?:\.\d+)?", raw)
    if not match:
        return 0.5
    return min(max(float(match.group()) / 10.0, 0.0), 1.0)


def verdict_from_score(score: float, *, thresholds: CragThresholds) -> CragVerdict:
    """Map a single aggregate score onto a verdict.

    Used by the adaptive path, which has one confidence number rather than a
    per-chunk breakdown.

    Example:
        >>> t = CragThresholds()
        >>> verdict_from_score(0.9, thresholds=t).value
        'correct'
        >>> verdict_from_score(0.5, thresholds=t).value
        'ambiguous'
        >>> verdict_from_score(0.1, thresholds=t).value
        'incorrect'
    """
    if score >= thresholds.correct_at_or_above:
        return CragVerdict.CORRECT
    if score < thresholds.incorrect_below:
        return CragVerdict.INCORRECT
    return CragVerdict.AMBIGUOUS


class TavilyWebSearch:
    """Web search fallback backed by Tavily.

    Results are marked ``RetrievalSource.WEB`` and carry no ``document_id``, so
    the citation UI can label them as external and the per-document cap does not
    lump unrelated pages together. Keeping that distinction visible matters: a
    user is entitled to know when an answer came from the open web rather than
    from their own documents.
    """

    def __init__(self, *, api_key: str, timeout: float = 12.0) -> None:
        """Create the client."""
        self._api_key = api_key
        self._timeout = timeout

    async def search(self, query: str, *, max_results: int = 5) -> list[RetrievedChunk]:
        """Search the web, returning results shaped as retrieval hits."""
        import httpx

        from src.retrieval.types import RetrievalSource

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self._api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "advanced",
                        "include_answer": False,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001 - fallback failure is not fatal
            log.warning("web search fallback failed", reason=str(exc))
            return []

        return [
            RetrievedChunk(
                chunk_id=f"web:{result.get('url', '')}",
                content=result.get("content", ""),
                score=float(result.get("score", 0.0)),
                source=RetrievalSource.WEB,
                document_title=result.get("title"),
                metadata={"url": result.get("url"), "external": True},
            )
            for result in payload.get("results", [])
            if result.get("content")
        ]


class NoWebSearch:
    """Web fallback for tenants with it disabled, and for offline tests.

    Returning nothing is the honest behaviour: the agent then tells the user it
    could not find an answer, which is correct, rather than answering from
    irrelevant context.
    """

    async def search(self, query: str, *, max_results: int = 5) -> list[RetrievedChunk]:
        """Return no results."""
        return []
