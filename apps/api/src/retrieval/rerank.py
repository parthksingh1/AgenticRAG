"""Cross-encoder reranking.

Retrieval and reranking answer different questions. A bi-encoder embeds the query
and the document separately, so it can index millions of chunks but never sees
the pair together. A cross-encoder reads query and passage jointly and scores the
actual relationship — far more accurate, far too slow to run over a corpus.

So: retrieve broadly and cheaply, then rerank the top few dozen precisely. This
is the single highest-leverage quality step in the pipeline, and it is also where
latency concentrates, which is why the implementations here batch, cap and time
themselves.

Three implementations:

* :class:`CrossEncoderReranker` — local ``BAAI/bge-reranker-v2-m3``, no network.
* :class:`CohereReranker` — the hosted Rerank v3 API, for tenants who prefer it.
* :class:`IdentityReranker` — passes scores through, so the eval harness can
  measure exactly what reranking is worth.

Example:
    >>> IdentityReranker().name
    'identity'
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from src.core.logging import get_logger
from src.retrieval.types import RetrievedChunk

log = get_logger(__name__)

#: Reranking cost is linear in candidates. Beyond this the latency is not worth
#: the marginal recall, and the candidates that far down rarely surface anyway.
MAX_RERANK_CANDIDATES = 60


class Reranker(ABC):
    """Rescores retrieved candidates against the query."""

    name: str

    @abstractmethod
    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        """Return the ``top_n`` chunks, reordered, each carrying a rerank score."""

    async def aclose(self) -> None:
        """Release resources. Overridden where there are any."""
        return


class IdentityReranker(Reranker):
    """Passes candidates through untouched.

    Exists so "reranking improves context precision by N points" is a measurable
    claim rather than an assumption: the eval harness runs the same pipeline with
    this swapped in.
    """

    name = "identity"

    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        """Truncate to ``top_n``, preserving the incoming order.

        Example:
            >>> import asyncio
            >>> from src.retrieval.types import RetrievalSource
            >>> hits = [
            ...     RetrievedChunk(chunk_id=str(i), content="c", score=1.0,
            ...                    source=RetrievalSource.FUSED)
            ...     for i in range(3)
            ... ]
            >>> reranked = asyncio.run(IdentityReranker().rerank("q", hits, top_n=2))
            >>> [c.chunk_id for c in reranked]
            ['0', '1']
        """
        return list(chunks[:top_n])


class CrossEncoderReranker(Reranker):
    """Local cross-encoder reranking with sentence-transformers.

    The model loads lazily and scores in a worker thread: a synchronous
    ``predict`` on the event loop would block every concurrent request on the
    worker for the duration, which at p95 is exactly when it hurts most.
    """

    name = "bge-reranker"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        """Configure the reranker without loading the model."""
        self.model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model: Any | None = None

    def _load(self) -> Any:
        """Load the cross-encoder on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading cross-encoder", model=self.model_name, device=self._device)
            self._model = CrossEncoder(
                self.model_name, device=self._device, max_length=self._max_length
            )
        return self._model

    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        """Score each (query, chunk) pair jointly and return the best ``top_n``."""
        import asyncio

        if not chunks:
            return []

        candidates = list(chunks[:MAX_RERANK_CANDIDATES])
        started = time.perf_counter()
        pairs = [(query, chunk.content) for chunk in candidates]

        try:
            model = self._load()
            scores = await asyncio.to_thread(
                model.predict, pairs, batch_size=self._batch_size, show_progress_bar=False
            )
        except Exception as exc:  # noqa: BLE001 - degrade rather than fail the turn
            log.error("cross-encoder rerank failed; keeping fusion order", reason=str(exc))
            return list(candidates[:top_n])

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.debug("reranked", candidates=len(candidates), latency_ms=elapsed_ms)

        return _apply_scores(candidates, [float(s) for s in scores], top_n=top_n)


class CohereReranker(Reranker):
    """Hosted reranking via the Cohere Rerank v3 API."""

    name = "cohere-rerank-v3"

    def __init__(self, *, api_key: str, model: str = "rerank-v3.5", timeout: float = 15.0) -> None:
        """Create the client."""
        import cohere

        self._client = cohere.AsyncClientV2(api_key=api_key, timeout=timeout)
        self._model = model

    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        """Call the Rerank API, falling back to the incoming order on failure."""
        if not chunks:
            return []

        candidates = list(chunks[:MAX_RERANK_CANDIDATES])
        try:
            response = await self._client.rerank(
                model=self._model,
                query=query,
                documents=[c.content for c in candidates],
                top_n=min(top_n, len(candidates)),
            )
        except Exception as exc:  # noqa: BLE001 - degrade rather than fail the turn
            log.error("cohere rerank failed; keeping fusion order", reason=str(exc))
            return list(candidates[:top_n])

        return [
            candidates[item.index].model_copy(update={"rerank_score": float(item.relevance_score)})
            for item in response.results
        ]

    async def aclose(self) -> None:
        """Close the HTTP client."""
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()


class ScriptedReranker(Reranker):
    """Reranker with a fixed score table, for deterministic tests."""

    name = "scripted"

    def __init__(self, scores: dict[str, float]) -> None:
        """Map chunk id to rerank score; unlisted chunks score zero."""
        self._scores = scores

    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        """Apply the scripted scores and reorder."""
        return _apply_scores(
            list(chunks), [self._scores.get(c.chunk_id, 0.0) for c in chunks], top_n=top_n
        )


def _apply_scores(
    chunks: Sequence[RetrievedChunk], scores: Sequence[float], *, top_n: int
) -> list[RetrievedChunk]:
    """Attach scores, sort descending and truncate.

    Ties break on chunk id so the output is deterministic, which matters because
    the eval harness snapshots retrieved source lists.

    Example:
        >>> from src.retrieval.types import RetrievalSource
        >>> a = RetrievedChunk(chunk_id="a", content="", score=0, source=RetrievalSource.FUSED)
        >>> b = RetrievedChunk(chunk_id="b", content="", score=0, source=RetrievalSource.FUSED)
        >>> [c.chunk_id for c in _apply_scores([a, b], [0.1, 0.9], top_n=2)]
        ['b', 'a']
    """
    scored = [
        chunk.model_copy(update={"rerank_score": score})
        for chunk, score in zip(chunks, scores, strict=True)
    ]
    scored.sort(key=lambda c: (-(c.rerank_score or 0.0), c.chunk_id))
    return scored[:top_n]
