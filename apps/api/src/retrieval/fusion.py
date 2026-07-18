"""Rank fusion.

Combining a dense list with a BM25 list is the single most consequential piece
of arithmetic in the retrieval stack, and the tempting approach — normalise the
scores and average them — is wrong. BM25 scores are unbounded and corpus
dependent; cosine similarities are bounded and cluster tightly near the top.
Min-max normalising them makes the fused ranking depend on the worst result in
each list, so adding one bad hit reorders the good ones.

Reciprocal Rank Fusion sidesteps this by discarding score magnitudes entirely
and combining *ranks*::

    score(d) = Σ_over_sources  weight_s / (k + rank_s(d))

Only ordering matters, so no calibration between backends is needed. ``k=60`` is
the constant from Cormack et al. (2009); a larger ``k`` flattens the curve and
lets agreement across many sources outweigh a single first-place hit.

:func:`weighted_score_fusion` is offered as the documented alternative for the
case where scores genuinely are comparable (two dense retrievers on the same
model), and the eval harness compares the two on the golden set.

Example:
    >>> dense = [
    ...     RetrievedChunk(chunk_id="a", content="A", score=0.9, source=RetrievalSource.DENSE)
    ... ]
    >>> sparse = [
    ...     RetrievedChunk(chunk_id="a", content="A", score=12.0, source=RetrievalSource.SPARSE)
    ... ]
    >>> fused = reciprocal_rank_fusion([dense, sparse], FusionConfig())
    >>> fused[0].chunk_id, fused[0].contributing_ranks
    ('a', {'dense': 1, 'sparse': 1})
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from src.retrieval.types import FusionConfig, RetrievalSource, RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Sequence[RetrievedChunk]],
    config: FusionConfig | None = None,
    *,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked result lists by reciprocal rank.

    Args:
        ranked_lists: One already-ordered list per retrieval source. Order within
            each list is authoritative; the scores are ignored by design.
        config: Fusion constant and per-source weights. Defaults to ``k=60`` with
            uniform weights.
        top_k: Truncate the fused list. ``None`` returns everything.

    Returns:
        Chunks ordered by descending fused score, each carrying ``fused_score``
        and the per-source ranks that produced it. Ties break on chunk id so the
        output is deterministic, which matters for snapshot evals.

    Example:
        >>> a = RetrievedChunk(chunk_id="a", content="A", score=1.0, source=RetrievalSource.DENSE)
        >>> b = RetrievedChunk(chunk_id="b", content="B", score=0.9, source=RetrievalSource.DENSE)
        >>> fused = reciprocal_rank_fusion([[a, b], [b, a]], FusionConfig(k=1))
        >>> [c.chunk_id for c in fused]
        ['a', 'b']
    """
    config = config or FusionConfig()

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    representatives: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        # A source that returns the same chunk twice must not double-count it.
        seen_in_list: set[str] = set()
        for position, chunk in enumerate(ranked, start=1):
            if chunk.chunk_id in seen_in_list:
                continue
            seen_in_list.add(chunk.chunk_id)

            source = chunk.source.value
            weight = config.weight_for(chunk.source)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (
                config.k + position
            )
            ranks.setdefault(chunk.chunk_id, {})[source] = position
            representatives.setdefault(chunk.chunk_id, chunk)

    fused = [
        representatives[chunk_id].model_copy(
            update={
                "fused_score": score,
                "contributing_ranks": ranks[chunk_id],
                "source": RetrievalSource.FUSED,
            }
        )
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda c: (-(c.fused_score or 0.0), c.chunk_id))
    return fused[:top_k] if top_k is not None else fused


def weighted_score_fusion(
    ranked_lists: Iterable[Sequence[RetrievedChunk]],
    config: FusionConfig | None = None,
    *,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse by min-max normalised score rather than by rank.

    The documented alternative to RRF. Appropriate only when the sources produce
    genuinely comparable scores; on a dense + BM25 mix it is measurably worse,
    which is the point of keeping it runnable in the eval harness rather than
    only asserting it in prose.

    Each list is normalised to [0, 1] independently. A list whose scores are all
    equal normalises to 1.0 throughout, because the alternative — dividing by a
    zero range — would silently produce NaN and poison the ranking.

    Example:
        >>> a = RetrievedChunk(chunk_id="a", content="A", score=10.0, source=RetrievalSource.SPARSE)
        >>> b = RetrievedChunk(chunk_id="b", content="B", score=0.0, source=RetrievalSource.SPARSE)
        >>> [c.chunk_id for c in weighted_score_fusion([[a, b]])]
        ['a', 'b']
    """
    config = config or FusionConfig()

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    representatives: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        normalised = _min_max_normalise([chunk.score for chunk in ranked])
        seen_in_list: set[str] = set()
        for position, (chunk, value) in enumerate(zip(ranked, normalised, strict=True), start=1):
            if chunk.chunk_id in seen_in_list:
                continue
            seen_in_list.add(chunk.chunk_id)

            weight = config.weight_for(chunk.source)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight * value
            ranks.setdefault(chunk.chunk_id, {})[chunk.source.value] = position
            representatives.setdefault(chunk.chunk_id, chunk)

    fused = [
        representatives[chunk_id].model_copy(
            update={
                "fused_score": score,
                "contributing_ranks": ranks[chunk_id],
                "source": RetrievalSource.FUSED,
            }
        )
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda c: (-(c.fused_score or 0.0), c.chunk_id))
    return fused[:top_k] if top_k is not None else fused


def _min_max_normalise(values: Sequence[float]) -> list[float]:
    """Scale values into [0, 1], mapping a degenerate range to all-ones.

    Example:
        >>> _min_max_normalise([0.0, 5.0, 10.0])
        [0.0, 0.5, 1.0]
        >>> _min_max_normalise([3.0, 3.0])
        [1.0, 1.0]
        >>> _min_max_normalise([])
        []
    """
    if not values:
        return []
    low, high = min(values), max(values)
    span = high - low
    if span == 0.0:
        return [1.0] * len(values)
    return [(value - low) / span for value in values]


def deduplicate_by_document(
    chunks: Sequence[RetrievedChunk], *, max_per_document: int
) -> list[RetrievedChunk]:
    """Cap how many chunks any one document may contribute, preserving order.

    Without this, a single long document reliably fills the entire context
    window, and the answer becomes confidently one-sided. Applied after fusion
    and before reranking.

    Args:
        chunks: Fused hits, already in the desired order.
        max_per_document: Ceiling per document. Must be at least 1.

    Example:
        >>> mk = lambda i, d: RetrievedChunk(
        ...     chunk_id=i, content=i, score=1.0, source=RetrievalSource.FUSED, document_id=d
        ... )
        >>> hits = [mk("a", "d1"), mk("b", "d1"), mk("c", "d2")]
        >>> [c.chunk_id for c in deduplicate_by_document(hits, max_per_document=1)]
        ['a', 'c']
    """
    if max_per_document < 1:
        msg = "max_per_document must be at least 1"
        raise ValueError(msg)

    counts: dict[str, int] = {}
    kept: list[RetrievedChunk] = []
    for chunk in chunks:
        # Chunks with no document id (web results) are never capped together.
        key = chunk.document_id or f"__unbound__{chunk.chunk_id}"
        if counts.get(key, 0) >= max_per_document:
            continue
        counts[key] = counts.get(key, 0) + 1
        kept.append(chunk)
    return kept
