"""Retriever protocol and shared filtering.

Every retrieval backend implements one method. That uniformity is what lets the
hybrid orchestrator add a backend without touching fusion, reranking, the agent
or the API — and what lets the eval harness swap a real backend for a fake one
and measure the rest of the pipeline in isolation.

Filters are applied by each backend in its own query language, but they are
*specified* once in :class:`~src.retrieval.types.RetrievalRequest`, so a filter
that works against Postgres behaves identically against OpenSearch.

Example:
    >>> InMemoryRetriever(chunks=[]).name
    'memory'
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

from src.core.logging import get_logger
from src.retrieval.types import RetrievalRequest, RetrievalSource, RetrievedChunk

log = get_logger(__name__)


class Retriever(ABC):
    """Finds candidate chunks for a query within one tenant."""

    #: Stable name, recorded on every retrieval log line.
    name: str
    #: Which source label the hits carry, so fusion can attribute and weight them.
    source: RetrievalSource

    @abstractmethod
    async def retrieve(self, request: RetrievalRequest, *, tenant_id: str) -> list[RetrievedChunk]:
        """Return candidates ordered best-first.

        Implementations must scope to ``tenant_id`` themselves. The ORM guard in
        :mod:`src.core.db` covers ORM queries, but OpenSearch, Neo4j and the
        vector index are outside it, so each backend carries the filter
        explicitly and is covered by the isolation test suite.
        """

    async def aclose(self) -> None:
        """Release any client resources. Overridden where there are any."""
        return


def matches_filters(
    request: RetrievalRequest,
    *,
    document_id: str | None,
    tags: Sequence[str] = (),
    kind: str | None = None,
    effective_date: datetime | None = None,
    is_stale: bool = False,
) -> bool:
    """Whether a candidate satisfies the request's filters.

    Used by backends that cannot express every filter natively (and by the
    in-memory retriever), so filter semantics are defined once rather than three
    times with subtle differences.

    Example:
        >>> req = RetrievalRequest(query="q", document_ids=("doc_1",))
        >>> matches_filters(req, document_id="doc_1")
        True
        >>> matches_filters(req, document_id="doc_2")
        False
    """
    if request.document_ids and document_id not in request.document_ids:
        return False
    if request.tags and not set(request.tags) & set(tags):
        return False
    if request.kinds and kind not in {k.value for k in request.kinds}:
        return False
    if is_stale and not request.include_stale:
        return False
    if effective_date is not None:
        if request.date_from and effective_date.isoformat() < request.date_from:
            return False
        if request.date_to and effective_date.isoformat() > request.date_to:
            return False
    return True


class InMemoryRetriever(Retriever):
    """Exact-match retriever over an in-process corpus.

    Not a toy: it is what the unit tests, the eval harness's offline mode and the
    chaos suite use to exercise everything *around* retrieval without a database.
    Scoring is deliberately crude (term overlap) because the point is to make the
    ranking predictable, not good.
    """

    name = "memory"
    source = RetrievalSource.DENSE

    def __init__(self, chunks: Sequence[RetrievedChunk], *, tenant_id: str = "ten_test") -> None:
        """Create a retriever over a fixed corpus owned by one tenant."""
        self._chunks = list(chunks)
        self._tenant_id = tenant_id

    async def retrieve(self, request: RetrievalRequest, *, tenant_id: str) -> list[RetrievedChunk]:
        """Score by term overlap and return the top k.

        Example:
            >>> import asyncio
            >>> corpus = [
            ...     RetrievedChunk(
            ...         chunk_id="a", content="retrieval augmented generation",
            ...         score=0.0, source=RetrievalSource.DENSE,
            ...     ),
            ...     RetrievedChunk(
            ...         chunk_id="b", content="unrelated text",
            ...         score=0.0, source=RetrievalSource.DENSE,
            ...     ),
            ... ]
            >>> r = InMemoryRetriever(corpus)
            >>> hits = asyncio.run(
            ...     r.retrieve(RetrievalRequest(query="retrieval"), tenant_id="ten_test")
            ... )
            >>> [h.chunk_id for h in hits]
            ['a']
        """
        if tenant_id != self._tenant_id:
            return []

        terms = {t.lower() for t in request.query.split()}
        for expansion in request.expansions:
            terms |= {t.lower() for t in expansion.split()}

        scored: list[RetrievedChunk] = []
        for chunk in self._chunks:
            if not matches_filters(request, document_id=chunk.document_id, kind=chunk.kind.value):
                continue
            words = {w.lower().strip(".,;:()[]") for w in chunk.content.split()}
            overlap = len(terms & words)
            if overlap:
                scored.append(chunk.model_copy(update={"score": overlap / max(len(terms), 1)}))

        scored.sort(key=lambda c: (-c.score, c.chunk_id))
        return scored[: request.top_k]
