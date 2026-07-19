"""Typed contracts for retrieval.

Every retriever — dense, sparse, ColBERT, graph, web fallback — returns the same
:class:`RetrievedChunk` shape, which is what lets fusion, reranking and the
citation binder be written once rather than once per backend.

Scores from different backends are not comparable (BM25 is unbounded, cosine is
in [-1, 1]), so :class:`RetrievedChunk` keeps the raw ``score`` alongside the
``source`` that produced it and never pretends they share a scale. Fusion is the
only place allowed to combine them, and it does so by rank, not by value.

Example:
    >>> hit = RetrievedChunk(
    ...     chunk_id="chk_1", content="text", score=0.8, source=RetrievalSource.DENSE
    ... )
    >>> hit.source.value
    'dense'
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.document import ChunkKind


class RetrievalSource(StrEnum):
    """Which backend produced a hit. Kept on every result for attribution."""

    DENSE = "dense"
    SPARSE = "sparse"
    COLBERT = "colbert"
    GRAPH = "graph"
    WEB = "web"
    CACHE = "cache"
    FUSED = "fused"


class CragVerdict(StrEnum):
    """Corrective-RAG assessment of whether retrieval actually answered the query."""

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class RetrievedChunk(BaseModel):
    """One retrieval hit, before fusion or reranking."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    content: str
    score: float = Field(
        description="Backend-native score. Not comparable across sources; fuse by rank."
    )
    source: RetrievalSource
    document_id: str | None = None
    document_title: str | None = None
    kind: ChunkKind = ChunkKind.PROSE
    ordinal: int | None = None
    page_number: int | None = None
    section_path: tuple[str, ...] = ()
    #: Set by the reranker. None means this hit has not been reranked.
    rerank_score: float | None = None
    #: Set by fusion. None means this hit has not been fused.
    fused_score: float | None = None
    #: Per-source ranks that contributed to ``fused_score``, for explainability
    #: in the admin failure explorer.
    contributing_ranks: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def effective_score(self) -> float:
        """The score a consumer should order by, newest signal winning.

        Example:
            >>> hit = RetrievedChunk(
            ...     chunk_id="c", content="t", score=0.2, source=RetrievalSource.DENSE
            ... )
            >>> hit.effective_score
            0.2
            >>> hit.model_copy(update={"fused_score": 0.5, "rerank_score": 0.9}).effective_score
            0.9
        """
        if self.rerank_score is not None:
            return self.rerank_score
        if self.fused_score is not None:
            return self.fused_score
        return self.score


class RetrievalRequest(BaseModel):
    """A single retrieval call, fully specified.

    Carrying the filters as typed fields rather than a free-form dict is what
    lets the SQL and OpenSearch backends share one validated contract, and what
    stops a caller from smuggling an unfiltered query past tenant scoping.
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=200)
    #: Additional rewritten forms (HyDE document, multi-query variants). The
    #: original query is always searched too.
    expansions: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    kinds: tuple[ChunkKind, ...] = ()
    #: Inclusive ISO-8601 date bounds on ``Document.effective_date``.
    date_from: str | None = None
    date_to: str | None = None
    include_stale: bool = False
    #: Multiplier applied to recent documents when the query is time-sensitive.
    recency_boost: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def all_queries(self) -> tuple[str, ...]:
        """The original query followed by its expansions, de-duplicated.

        Example:
            >>> RetrievalRequest(query="a", expansions=("b", "a")).all_queries
            ('a', 'b')
        """
        seen: dict[str, None] = {self.query: None}
        for expansion in self.expansions:
            seen.setdefault(expansion, None)
        return tuple(seen)


class RetrievalResult(BaseModel):
    """The outcome of one retrieval stage, with the telemetry it produced."""

    model_config = ConfigDict(frozen=True)

    chunks: tuple[RetrievedChunk, ...]
    strategy: str
    latency_ms: int = 0
    #: Per-source latencies, so the dashboard can attribute a slow retrieval to
    #: the backend actually responsible rather than to "retrieval".
    source_latencies_ms: dict[str, int] = Field(default_factory=dict)
    expanded: bool = False
    crag_verdict: CragVerdict | None = None
    web_fallback_used: bool = False

    @property
    def top_score(self) -> float | None:
        """Highest effective score, or None when nothing was retrieved."""
        if not self.chunks:
            return None
        return max(chunk.effective_score for chunk in self.chunks)

    @property
    def mean_score(self) -> float | None:
        """Mean effective score, or None when nothing was retrieved."""
        if not self.chunks:
            return None
        return sum(chunk.effective_score for chunk in self.chunks) / len(self.chunks)

    def top(self, n: int) -> tuple[RetrievedChunk, ...]:
        """The ``n`` highest-scoring chunks, in descending order."""
        return tuple(sorted(self.chunks, key=lambda c: -c.effective_score)[:n])


class FusionConfig(BaseModel):
    """How to combine ranked lists from multiple retrieval backends."""

    model_config = ConfigDict(frozen=True)

    #: RRF smoothing constant. 60 is the value from Cormack et al. (2009) and the
    #: default everywhere; lowering it sharpens the preference for rank-1 hits.
    k: int = Field(default=60, ge=1, le=1000)
    #: Per-source multipliers. A source absent from this map contributes with
    #: weight 1.0, so adding a backend never silently changes existing behaviour.
    weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_weights(self) -> Self:
        """Reject negative weights, which would invert a source's contribution."""
        negative = {name: w for name, w in self.weights.items() if w < 0}
        if negative:
            msg = f"fusion weights must be non-negative, got {negative}"
            raise ValueError(msg)
        return self

    def weight_for(self, source: RetrievalSource | str) -> float:
        """Weight for a source, defaulting to 1.0.

        Example:
            >>> FusionConfig(weights={"dense": 2.0}).weight_for(RetrievalSource.DENSE)
            2.0
            >>> FusionConfig().weight_for("sparse")
            1.0
        """
        key = source.value if isinstance(source, RetrievalSource) else source
        return self.weights.get(key, 1.0)
