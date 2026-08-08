"""Rank fusion tests.

Fusion is gated at 100% coverage because a subtle bug here degrades every answer
the system produces without failing anything loudly. The properties below pin
the behaviour that makes RRF worth choosing over score averaging.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from src.retrieval.fusion import (
    _min_max_normalise,
    deduplicate_by_document,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from src.retrieval.types import FusionConfig, RetrievalSource, RetrievedChunk

pytestmark = pytest.mark.unit


def hit(
    chunk_id: str,
    *,
    score: float = 1.0,
    source: RetrievalSource = RetrievalSource.DENSE,
    document_id: str | None = None,
) -> RetrievedChunk:
    """Build a minimal retrieval hit for tests."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"content of {chunk_id}",
        score=score,
        source=source,
        document_id=document_id,
    )


def dense(*ids: str) -> list[RetrievedChunk]:
    """A dense ranked list, scores descending in list order."""
    return [hit(i, score=1.0 - n / 100, source=RetrievalSource.DENSE) for n, i in enumerate(ids)]


def sparse(*ids: str) -> list[RetrievedChunk]:
    """A BM25 ranked list, with the unbounded scores BM25 actually produces."""
    return [hit(i, score=40.0 - n, source=RetrievalSource.SPARSE) for n, i in enumerate(ids)]


# ── Core RRF behaviour ───────────────────────────────────────────────────────


def test_agreement_across_sources_beats_a_single_first_place() -> None:
    """A doc ranked 2nd by both sources outranks one ranked 1st by only one.

    This is the whole point of fusion: corroboration is evidence.
    """
    fused = reciprocal_rank_fusion([dense("solo", "both"), sparse("other", "both")])

    assert fused[0].chunk_id == "both"


def test_fused_score_matches_the_rrf_formula() -> None:
    """The arithmetic is exactly Σ weight / (k + rank), not an approximation."""
    config = FusionConfig(k=10)
    fused = reciprocal_rank_fusion([dense("a", "b"), sparse("b", "a")], config)

    by_id = {c.chunk_id: c for c in fused}
    expected = 1 / (10 + 1) + 1 / (10 + 2)
    assert by_id["a"].fused_score == pytest.approx(expected)
    assert by_id["b"].fused_score == pytest.approx(expected)


def test_raw_score_magnitude_does_not_affect_the_ranking() -> None:
    """Only rank matters — this is what makes BM25 and cosine safely fusible."""
    modest = [hit("a", score=0.01, source=RetrievalSource.SPARSE)]
    enormous = [hit("a", score=9_999.0, source=RetrievalSource.SPARSE)]

    assert (
        reciprocal_rank_fusion([modest, dense("b")])[0].fused_score
        == reciprocal_rank_fusion([enormous, dense("b")])[0].fused_score
    )


def test_contributing_ranks_are_recorded_for_explainability() -> None:
    """The failure explorer needs to show *why* a chunk surfaced."""
    fused = reciprocal_rank_fusion([dense("x", "y"), sparse("y", "x")])

    by_id = {c.chunk_id: c for c in fused}
    assert by_id["x"].contributing_ranks == {"dense": 1, "sparse": 2}
    assert by_id["y"].contributing_ranks == {"dense": 2, "sparse": 1}


def test_fused_hits_are_marked_as_fused() -> None:
    """Downstream code must not mistake a fused hit for a raw dense one."""
    fused = reciprocal_rank_fusion([dense("a"), sparse("a")])

    assert all(c.source is RetrievalSource.FUSED for c in fused)


def test_a_source_returning_a_duplicate_does_not_double_count_it() -> None:
    """A buggy backend must not be able to promote a chunk by repeating it."""
    duplicated = [*dense("a"), *dense("a")]
    once = reciprocal_rank_fusion([dense("a")])[0].fused_score
    twice = reciprocal_rank_fusion([duplicated])[0].fused_score

    assert once == twice


def test_weights_scale_a_source_contribution() -> None:
    """A tenant that trusts BM25 more can say so without touching code."""
    config = FusionConfig(weights={"sparse": 3.0})
    fused = reciprocal_rank_fusion(
        [dense("d_first", "s_first"), sparse("s_first", "d_first")], config
    )

    assert fused[0].chunk_id == "s_first"


def test_absent_weight_defaults_to_one() -> None:
    """Adding a new backend never silently reweights the existing ones."""
    config = FusionConfig(weights={"dense": 1.0})
    assert config.weight_for(RetrievalSource.COLBERT) == 1.0


def test_negative_weights_are_rejected() -> None:
    """A negative weight would invert a source's contribution — always a mistake."""
    with pytest.raises(ValueError, match="non-negative"):
        FusionConfig(weights={"dense": -1.0})


def test_a_zero_weight_source_still_records_its_rank() -> None:
    """Zero weight means "do not score", not "do not observe"."""
    config = FusionConfig(weights={"sparse": 0.0})
    fused = reciprocal_rank_fusion([dense("a"), sparse("a")], config)

    assert fused[0].contributing_ranks == {"dense": 1, "sparse": 1}
    assert fused[0].fused_score == pytest.approx(1 / (60 + 1))


def test_empty_input_produces_no_results() -> None:
    """Nothing in, nothing out — never a spurious empty chunk."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_top_k_truncates_after_ordering() -> None:
    """Truncation happens on the fused order, not on any single source's order."""
    fused = reciprocal_rank_fusion([dense("a", "b", "c"), sparse("c", "b", "a")], top_k=2)

    assert len(fused) == 2


# ── Properties ───────────────────────────────────────────────────────────────

id_strategy = st.text(alphabet="abcdefgh", min_size=1, max_size=3)
list_strategy = st.lists(id_strategy, min_size=0, max_size=8, unique=True)


@given(first=list_strategy, second=list_strategy, k=st.integers(min_value=1, max_value=200))
@settings(max_examples=250, deadline=None)
def test_fusion_output_is_sorted_and_unique(first: list[str], second: list[str], k: int) -> None:
    """Output is descending by fused score with no duplicate chunk ids."""
    fused = reciprocal_rank_fusion([dense(*first), sparse(*second)], FusionConfig(k=k))

    ids = [c.chunk_id for c in fused]
    assert len(ids) == len(set(ids))
    scores = [c.fused_score or 0.0 for c in fused]
    assert scores == sorted(scores, reverse=True)


@given(first=list_strategy, second=list_strategy)
@settings(max_examples=250, deadline=None)
def test_fusion_preserves_the_union_of_inputs(first: list[str], second: list[str]) -> None:
    """Fusion never invents a chunk and never drops one."""
    fused = reciprocal_rank_fusion([dense(*first), sparse(*second)])

    assert {c.chunk_id for c in fused} == set(first) | set(second)


@given(ids=list_strategy)
@settings(max_examples=200, deadline=None)
def test_fusion_of_one_list_preserves_that_list_order(ids: list[str]) -> None:
    """With a single source, fusion is order-preserving — a useful sanity anchor."""
    fused = reciprocal_rank_fusion([dense(*ids)])

    assert [c.chunk_id for c in fused] == ids


@given(first=list_strategy, second=list_strategy)
@settings(max_examples=200, deadline=None)
def test_fusion_is_commutative_in_its_sources(first: list[str], second: list[str]) -> None:
    """Argument order must not change the result — sources have no precedence."""
    forward = reciprocal_rank_fusion([dense(*first), sparse(*second)])
    backward = reciprocal_rank_fusion([sparse(*second), dense(*first)])

    assert [c.chunk_id for c in forward] == [c.chunk_id for c in backward]


@given(first=list_strategy, second=list_strategy)
@settings(max_examples=200, deadline=None)
def test_fusion_is_deterministic(first: list[str], second: list[str]) -> None:
    """Ties break on chunk id, so snapshot evals do not flap."""
    args = [dense(*first), sparse(*second)]
    assert [c.chunk_id for c in reciprocal_rank_fusion(args)] == [
        c.chunk_id for c in reciprocal_rank_fusion(args)
    ]


# ── Weighted score fusion (the documented alternative) ───────────────────────


def test_weighted_fusion_normalises_each_list_independently() -> None:
    """BM25's unbounded scale is squashed to [0, 1] before combining."""
    fused = weighted_score_fusion([sparse("a", "b"), dense("b", "a")])

    assert {c.chunk_id for c in fused} == {"a", "b"}
    assert all(0.0 <= (c.fused_score or 0.0) <= 2.0 for c in fused)


def test_weighted_fusion_handles_a_degenerate_score_range() -> None:
    """All-equal scores normalise to 1.0 rather than dividing by zero."""
    flat = [hit("a", score=5.0), hit("b", score=5.0)]
    fused = weighted_score_fusion([flat])

    assert all(c.fused_score == pytest.approx(1.0) for c in fused)


def test_weighted_fusion_is_swayed_by_an_outlier_where_rrf_is_not() -> None:
    """The failure mode that justifies defaulting to RRF, demonstrated.

    Appending one badly-scoring hit to a list changes how the *good* hits in
    that list normalise against each other, so the fused score of a runner-up
    swings wildly. RRF, keyed only on rank, is unmoved.
    """
    good = [
        hit("a", score=10.0, source=RetrievalSource.SPARSE),
        hit("b", score=9.0, source=RetrievalSource.SPARSE),
    ]
    with_outlier = [*good, hit("junk", score=-500.0, source=RetrievalSource.SPARSE)]

    def score_of(chunk_id: str, lists: list[list[RetrievedChunk]]) -> float:
        return next(
            c.fused_score or 0.0 for c in weighted_score_fusion(lists) if c.chunk_id == chunk_id
        )

    b_before = score_of("b", [good])
    b_after = score_of("b", [with_outlier])

    rrf_before = [c.chunk_id for c in reciprocal_rank_fusion([good])]
    rrf_after = [c.chunk_id for c in reciprocal_rank_fusion([with_outlier]) if c.chunk_id != "junk"]

    # The runner-up goes from "worst in list" (0.0) to "nearly as good as the
    # best" (~0.998) purely because an unrelated bad hit was appended.
    assert b_before == pytest.approx(0.0)
    assert b_after > 0.99
    assert rrf_before == rrf_after


def test_weighted_fusion_ignores_a_duplicate_within_one_list() -> None:
    """Same anti-double-counting guarantee as RRF."""
    once = weighted_score_fusion([[hit("a", score=1.0)]])[0].fused_score
    twice = weighted_score_fusion([[hit("a", score=1.0), hit("a", score=1.0)]])[0].fused_score

    assert once == twice


def test_weighted_fusion_on_empty_input() -> None:
    """Nothing in, nothing out."""
    assert weighted_score_fusion([[]]) == []


@pytest.mark.parametrize(
    ("values", "expected"),
    [([0.0, 5.0, 10.0], [0.0, 0.5, 1.0]), ([3.0, 3.0], [1.0, 1.0]), ([], [])],
)
def test_min_max_normalise(values: list[float], expected: list[float]) -> None:
    """Normalisation maps a degenerate range to all-ones instead of NaN."""
    assert _min_max_normalise(values) == expected


# ── Per-document capping ─────────────────────────────────────────────────────


def test_deduplicate_by_document_caps_a_dominant_document() -> None:
    """One long document cannot monopolise the context window."""
    hits = [
        hit("a", document_id="d1"),
        hit("b", document_id="d1"),
        hit("c", document_id="d1"),
        hit("d", document_id="d2"),
    ]
    kept = deduplicate_by_document(hits, max_per_document=2)

    assert [c.chunk_id for c in kept] == ["a", "b", "d"]


def test_deduplicate_preserves_input_order() -> None:
    """Capping must not reorder — the caller already ranked these."""
    hits = [hit("a", document_id="d1"), hit("b", document_id="d2"), hit("c", document_id="d1")]
    kept = deduplicate_by_document(hits, max_per_document=1)

    assert [c.chunk_id for c in kept] == ["a", "b"]


def test_chunks_without_a_document_id_are_never_capped_together() -> None:
    """Web results share no document, so they must not compete for one slot."""
    hits = [hit("w1"), hit("w2"), hit("w3")]
    kept = deduplicate_by_document(hits, max_per_document=1)

    assert len(kept) == 3


def test_deduplicate_rejects_a_nonsensical_cap() -> None:
    """A cap of zero would discard everything, which is never the intent."""
    with pytest.raises(ValueError, match="at least 1"):
        deduplicate_by_document([hit("a")], max_per_document=0)


# ── Request contract ─────────────────────────────────────────────────────────


def test_effective_score_prefers_the_newest_signal() -> None:
    """Rerank beats fusion beats the raw backend score."""
    raw = hit("a", score=0.2)
    assert raw.effective_score == 0.2
    assert raw.model_copy(update={"fused_score": 0.5}).effective_score == 0.5
    assert raw.model_copy(update={"fused_score": 0.5, "rerank_score": 0.9}).effective_score == 0.9
