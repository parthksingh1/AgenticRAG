"""Contextual Retrieval.

Anthropic's technique: before embedding a chunk, ask a cheap model to write one
or two sentences situating that chunk inside its document, and prepend them to
the text that gets embedded. The chunk that reads "revenue grew 12%" becomes
"This section of the ACME 2024 annual report discusses Q3 segment performance.
Revenue grew 12%." — which is retrievable by a query about ACME's Q3 revenue,
where the bare chunk was not.

Two details matter for this to be worth its cost:

* The preamble is **embedded but never displayed.** The user sees the original
  chunk as their citation; showing generated text as if it were source material
  would be a quiet form of fabrication.
* The document context is sent as a **cached prefix**. Every chunk in a document
  shares the same document summary, so with prompt caching the marginal cost per
  chunk is small rather than "re-read the whole document once per chunk".

Failures are non-fatal by design: a chunk that cannot be contextualised is
embedded as-is, because degraded retrieval beats a failed ingestion.

Example:
    >>> ContextualEnricher.PROMPT_NAME
    'contextual_retrieval'
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from src.core.logging import get_logger
from src.ingestion.types import ChunkDraft
from src.services.llm.router import LLMRouter
from src.services.llm.types import CompletionRequest, Message

log = get_logger(__name__)

#: How much of the document to show the model as context. Beyond this the cost
#: of the cached prefix stops paying for itself.
MAX_DOCUMENT_CONTEXT_CHARS = 40_000

_SYSTEM = (
    "You situate an excerpt within its source document so it can be retrieved on "
    "its own. Reply with one or two short sentences of context and nothing else. "
    "Do not summarise the excerpt, do not add facts that are not in the document, "
    "and do not use phrases like 'this chunk' or 'this excerpt'."
)

_USER = """<document>
{document}
</document>

Here is the excerpt to situate:

<excerpt>
{chunk}
</excerpt>

Give the short context that situates this excerpt within the document."""


class ContextualEnricher:
    """Generates situating context for chunks before they are embedded."""

    PROMPT_NAME = "contextual_retrieval"

    def __init__(
        self,
        *,
        router: LLMRouter,
        model: str | None = None,
        max_concurrency: int = 8,
        max_context_tokens: int = 120,
    ) -> None:
        """Create an enricher.

        Args:
            router: Router used for the generation calls.
            model: Model to use. Defaults to the tenant's cheap model, because
                this runs once per chunk and does not need a frontier model.
            max_concurrency: Simultaneous in-flight calls. Bounded so ingesting
                a large document cannot exhaust the tenant's rate limit and
                starve interactive chat traffic.
            max_context_tokens: Ceiling on the generated preamble.
        """
        self._router = router
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_context_tokens = max_context_tokens
        self.generated = 0
        self.failed = 0

    async def enrich(
        self,
        drafts: Sequence[ChunkDraft],
        *,
        document_text: str,
        document_title: str | None = None,
    ) -> list[ChunkDraft]:
        r"""Return copies of ``drafts`` carrying generated context preambles.

        Chunks whose enrichment fails are returned unchanged, so a partial
        outage degrades retrieval quality rather than failing the ingestion.

        Example:
            >>> import asyncio
            >>> from src.services.llm.providers import FakeProvider
            >>> from src.services.llm.router import LLMRouter, ModelPolicy
            >>> from src.services.llm import pricing
            >>> pricing.MODEL_PROVIDERS["ctx-model"] = "fake"
            >>> router = LLMRouter(
            ...     providers={"fake": FakeProvider(responses=["From the ACME report."])},
            ...     policy=ModelPolicy(default_model="ctx-model"),
            ... )
            >>> drafts = [ChunkDraft(content="Revenue grew 12%.", ordinal=0)]
            >>> out = asyncio.run(
            ...     ContextualEnricher(router=router).enrich(drafts, document_text="...")
            ... )
            >>> out[0].embedded_text
            'From the ACME report.\n\nRevenue grew 12%.'
            >>> out[0].content
            'Revenue grew 12%.'
        """
        if not drafts:
            return []

        document = _prepare_document(document_text, title=document_title)
        results = await asyncio.gather(
            *(self._context_for(draft, document) for draft in drafts),
            return_exceptions=False,
        )
        return [
            draft if context is None else draft.model_copy(update={"context_preamble": context})
            for draft, context in zip(drafts, results, strict=True)
        ]

    async def _context_for(self, draft: ChunkDraft, document: str) -> str | None:
        """Generate one chunk's context, or None if it could not be produced."""
        async with self._semaphore:
            request = CompletionRequest(
                messages=(
                    Message.system(_SYSTEM),
                    Message.user(_USER.format(document=document, chunk=draft.content)),
                ),
                model=self._model or "",
                max_tokens=self._max_context_tokens,
                temperature=0.0,
                node="contextual_retrieval",
            )
            try:
                completion = await self._router.complete(
                    request.model_copy(update={"model": self._model or request.model}),
                    allow_fallback=False,
                )
            except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
                self.failed += 1
                log.warning(
                    "contextual enrichment failed; embedding chunk without context",
                    ordinal=draft.ordinal,
                    reason=str(exc),
                )
                return None

        context = completion.content.strip()
        if not context:
            self.failed += 1
            return None
        self.generated += 1
        return context


def _prepare_document(text: str, *, title: str | None) -> str:
    """Trim the document to the cacheable context window, keeping both ends.

    Keeping the head and the tail rather than a prefix matters: the head carries
    the title, abstract and framing, while the tail often carries conclusions and
    defined terms. A middle-out trim preserves both.

    Example:
        >>> _prepare_document("abcdefghij", title="T")[:2]
        'T:'
    """
    body = text.strip()
    if len(body) > MAX_DOCUMENT_CONTEXT_CHARS:
        half = MAX_DOCUMENT_CONTEXT_CHARS // 2
        body = f"{body[:half]}\n\n[...]\n\n{body[-half:]}"
    return f"{title}:\n\n{body}" if title else body
