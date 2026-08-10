"""Retrieval orchestration tests.

The behaviours pinned here are the ones that decide whether a bad day is a
slightly worse answer or a failed conversation: partial backend failure, bounded
adaptive widening, and the CRAG routes that stop the generator answering from
irrelevant context.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from src.retrieval.base import InMemoryRetriever, Retriever, matches_filters
from src.retrieval.corrective import (
    CragThresholds,
    NoWebSearch,
    RetrievalEvaluator,
    parse_score,
    verdict_from_score,
)
from src.retrieval.graph import heuristic_entities, normalise_relation
from src.retrieval.hybrid import HybridConfig, HybridRetriever, _low_confidence
from src.retrieval.rerank import IdentityReranker, ScriptedReranker
from src.retrieval.rewrite import (
    QueryRewriter,
    _parse_lines,
    looks_multi_hop,
    looks_time_sensitive,
)
from src.retrieval.sparse import index_name_for
from src.retrieval.types import (
    CragVerdict,
    RetrievalRequest,
    RetrievalSource,
    RetrievedChunk,
)
from src.services.llm import pricing
from src.services.llm.providers import FakeProvider
from src.services.llm.router import LLMRouter, ModelPolicy

pytestmark = pytest.mark.unit

TENANT = "ten_test"


def chunk(
    chunk_id: str,
    *,
    content: str = "content",
    score: float = 1.0,
    source: RetrievalSource = RetrievalSource.DENSE,
    document_id: str | None = None,
) -> RetrievedChunk:
    """Build a retrieval hit for tests."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        score=score,
        source=source,
        document_id=document_id,
    )


class StubRetriever(Retriever):
    """Returns a fixed list, or raises, to exercise the orchestrator."""

    def __init__(
        self,
        name: str,
        source: RetrievalSource,
        hits: Sequence[RetrievedChunk] = (),
        *,
        fail: bool = False,
    ) -> None:
        """Configure the stub."""
        self.name = name
        self.source = source
        self._hits = list(hits)
        self._fail = fail
        self.calls: list[RetrievalRequest] = []

    async def retrieve(self, request: RetrievalRequest, *, tenant_id: str) -> list[RetrievedChunk]:
        """Record the call and return the scripted hits."""
        self.calls.append(request)
        if self._fail:
            msg = "backend exploded"
            raise RuntimeError(msg)
        return list(self._hits)


def make_router(responses: Sequence[str]) -> LLMRouter:
    """A router over a fake provider returning scripted text."""
    pricing.MODEL_PROVIDERS["retrieval-test-model"] = "fake"
    return LLMRouter(
        providers={"fake": FakeProvider(responses=list(responses))},
        policy=ModelPolicy(default_model="retrieval-test-model"),
    )


# ── Filters ──────────────────────────────────────────────────────────────────


def test_document_filter_excludes_other_documents() -> None:
    """Scoped search must not reach outside the chosen documents."""
    request = RetrievalRequest(query="q", document_ids=("doc_1",))

    assert matches_filters(request, document_id="doc_1") is True
    assert matches_filters(request, document_id="doc_2") is False


def test_stale_chunks_are_hidden_unless_explicitly_requested() -> None:
    """Superseded chunks stay resolvable for old citations but out of new answers."""
    request = RetrievalRequest(query="q")

    assert matches_filters(request, document_id="d", is_stale=True) is False
    assert (
        matches_filters(
            RetrievalRequest(query="q", include_stale=True), document_id="d", is_stale=True
        )
        is True
    )


def test_expansions_are_deduplicated_against_the_original() -> None:
    """A rewrite that reproduces the query must not double its weight in fusion."""
    assert RetrievalRequest(query="a", expansions=("b", "a")).all_queries == ("a", "b")


# ── Orchestration ────────────────────────────────────────────────────────────


async def test_backends_are_fused_not_concatenated() -> None:
    """A chunk found by both backends must outrank one found by only one."""
    dense = StubRetriever("dense", RetrievalSource.DENSE, [chunk("solo"), chunk("both")])
    sparse = StubRetriever(
        "sparse",
        RetrievalSource.SPARSE,
        [
            chunk("other", source=RetrievalSource.SPARSE),
            chunk("both", source=RetrievalSource.SPARSE),
        ],
    )
    retriever = HybridRetriever(retrievers=[dense, sparse], config=HybridConfig(use_rerank=False))

    result = await retriever.retrieve(RetrievalRequest(query="q"), tenant_id=TENANT)

    assert result.chunks[0].chunk_id == "both"


async def test_one_backend_failing_does_not_fail_the_turn() -> None:
    """An OpenSearch outage must degrade the answer, not break chat."""
    dense = StubRetriever("dense", RetrievalSource.DENSE, [chunk("a")])
    sparse = StubRetriever("sparse", RetrievalSource.SPARSE, fail=True)
    retriever = HybridRetriever(retrievers=[dense, sparse], config=HybridConfig(use_rerank=False))

    result = await retriever.retrieve(RetrievalRequest(query="q"), tenant_id=TENANT)

    assert [c.chunk_id for c in result.chunks] == ["a"]


async def test_every_backend_failing_returns_empty_rather_than_raising() -> None:
    """The agent then says it could not find anything, which is the honest answer."""
    retriever = HybridRetriever(
        retrievers=[
            StubRetriever("dense", RetrievalSource.DENSE, fail=True),
            StubRetriever("sparse", RetrievalSource.SPARSE, fail=True),
        ],
        config=HybridConfig(use_rerank=False),
    )

    result = await retriever.retrieve(RetrievalRequest(query="q"), tenant_id=TENANT)

    assert result.chunks == ()


async def test_disabled_backends_are_not_called() -> None:
    """A tenant on dense-only must not pay OpenSearch latency."""
    sparse = StubRetriever("sparse", RetrievalSource.SPARSE, [chunk("s")])
    retriever = HybridRetriever(
        retrievers=[StubRetriever("dense", RetrievalSource.DENSE, [chunk("d")]), sparse],
        config=HybridConfig(use_sparse=False, use_rerank=False),
    )

    await retriever.retrieve(RetrievalRequest(query="q"), tenant_id=TENANT)

    assert sparse.calls == []


async def test_one_document_cannot_monopolise_the_context() -> None:
    """Without capping, a long document fills the window and the answer goes one-sided."""
    hits = [chunk(f"c{i}", document_id="doc_long") for i in range(5)]
    hits.append(chunk("other", document_id="doc_other"))
    retriever = HybridRetriever(
        retrievers=[StubRetriever("dense", RetrievalSource.DENSE, hits)],
        config=HybridConfig(use_rerank=False, max_per_document=2, top_k=10),
    )

    result = await retriever.retrieve(RetrievalRequest(query="q"), tenant_id=TENANT)

    from_long = [c for c in result.chunks if c.document_id == "doc_long"]
    assert len(from_long) == 2
    assert any(c.document_id == "doc_other" for c in result.chunks)


async def test_reranker_decides_the_final_order() -> None:
    """Reranking is the last word: fusion order must not survive it."""
    retriever = HybridRetriever(
        retrievers=[StubRetriever("dense", RetrievalSource.DENSE, [chunk("a"), chunk("b")])],
        config=HybridConfig(rerank_top_n=2),
        reranker=ScriptedReranker({"a": 0.1, "b": 0.9}),
    )

    result = await retriever.retrieve(RetrievalRequest(query="q"), tenant_id=TENANT)

    assert [c.chunk_id for c in result.chunks] == ["b", "a"]


async def test_strategy_label_records_what_actually_ran() -> None:
    """A regression must be traceable to a configuration change, not guessed at."""
    retriever = HybridRetriever(
        retrievers=[StubRetriever("dense", RetrievalSource.DENSE, [chunk("a")])],
        config=HybridConfig(use_sparse=False, use_corrective=True, use_rerank=False),
    )

    result = await retriever.retrieve(RetrievalRequest(query="q"), tenant_id=TENANT)

    assert result.strategy == "corrective+dense"


async def test_per_backend_latencies_are_reported() -> None:
    """Attributing a slow retrieval to "retrieval" is useless for debugging."""
    retriever = HybridRetriever(
        retrievers=[
            StubRetriever("dense", RetrievalSource.DENSE, [chunk("a")]),
            StubRetriever("sparse", RetrievalSource.SPARSE, [chunk("b")]),
        ],
        config=HybridConfig(use_rerank=False),
    )

    result = await retriever.retrieve(RetrievalRequest(query="q"), tenant_id=TENANT)

    assert set(result.source_latencies_ms) == {"dense", "sparse"}


# ── Adaptive widening ────────────────────────────────────────────────────────


async def test_weak_retrieval_triggers_one_wider_attempt() -> None:
    """Low confidence should widen k rather than answering from nothing."""
    weak = StubRetriever("dense", RetrievalSource.DENSE, [chunk("a", score=0.1)])
    retriever = HybridRetriever(
        retrievers=[weak], config=HybridConfig(use_adaptive=True, use_rerank=False, top_k=5)
    )

    result = await retriever.retrieve(RetrievalRequest(query="q", top_k=5), tenant_id=TENANT)

    assert result.expanded is True
    assert len(weak.calls) == 2
    assert weak.calls[1].top_k > weak.calls[0].top_k


async def test_widening_happens_at_most_once() -> None:
    """An unbounded confidence loop is how one question becomes a timeout."""
    weak = StubRetriever("dense", RetrievalSource.DENSE, [chunk("a", score=0.01)])
    retriever = HybridRetriever(
        retrievers=[weak], config=HybridConfig(use_adaptive=True, use_rerank=False)
    )

    await retriever.retrieve(RetrievalRequest(query="q"), tenant_id=TENANT)

    assert len(weak.calls) == 2


async def test_strong_retrieval_does_not_widen() -> None:
    """Widening a good result set only costs latency."""
    strong = StubRetriever(
        "dense", RetrievalSource.DENSE, [chunk(f"c{i}", score=0.95) for i in range(5)]
    )
    retriever = HybridRetriever(
        retrievers=[strong], config=HybridConfig(use_adaptive=True, use_rerank=False)
    )

    result = await retriever.retrieve(RetrievalRequest(query="q"), tenant_id=TENANT)

    assert result.expanded is False
    assert len(strong.calls) == 1


@pytest.mark.parametrize(
    ("hits", "expected"),
    [
        ([], True),
        ([chunk("a", score=0.1)], True),
        ([chunk(f"c{i}", score=0.9) for i in range(5)], False),
    ],
)
def test_low_confidence_heuristic(hits: list[RetrievedChunk], expected: bool) -> None:
    """Too few results, or uniformly weak ones, justify a wider search."""
    assert _low_confidence(hits, 5) is expected


# ── Corrective RAG ───────────────────────────────────────────────────────────


async def test_relevant_context_is_graded_correct() -> None:
    """A good retrieval must not be second-guessed into a web search."""
    evaluator = RetrievalEvaluator(router=make_router(["9"]))

    assessment = await evaluator.evaluate("q", [chunk("a"), chunk("b")])

    assert assessment.verdict is CragVerdict.CORRECT
    assert assessment.needs_web_search is False


async def test_irrelevant_context_routes_to_web_search() -> None:
    """This is the node's whole purpose: stop the generator inventing from noise."""
    evaluator = RetrievalEvaluator(router=make_router(["0"]))

    assessment = await evaluator.evaluate("q", [chunk("a")])

    assert assessment.verdict is CragVerdict.INCORRECT
    assert assessment.needs_web_search is True


async def test_ambiguous_context_is_kept_and_widened() -> None:
    """Discarding partially-relevant context loses the signal that made it ambiguous."""
    evaluator = RetrievalEvaluator(router=make_router(["5"]))

    assessment = await evaluator.evaluate("q", [chunk("a")])

    assert assessment.verdict is CragVerdict.AMBIGUOUS
    assert assessment.needs_wider_retrieval is True
    assert [c.chunk_id for c in assessment.kept] == ["a"]


async def test_empty_retrieval_is_incorrect_without_spending_anything() -> None:
    """There is nothing to grade, and the verdict is obvious."""
    router = make_router(["9"])
    evaluator = RetrievalEvaluator(router=router)

    assessment = await evaluator.evaluate("q", [])

    assert assessment.verdict is CragVerdict.INCORRECT
    assert assessment.graded_with == "empty"
    assert router.stats.calls == 0


async def test_grading_failure_scores_neutral_not_zero() -> None:
    """Treating an outage as "irrelevant" would route every turn to web search."""
    pricing.MODEL_PROVIDERS["crag-dead-model"] = "fake"
    router = LLMRouter(
        providers={"fake": FakeProvider(fail_times=99)},
        policy=ModelPolicy(default_model="crag-dead-model"),
    )
    evaluator = RetrievalEvaluator(router=router)

    assessment = await evaluator.evaluate("q", [chunk("a")])

    assert assessment.verdict is CragVerdict.AMBIGUOUS


async def test_web_fallback_results_are_marked_external() -> None:
    """Users are entitled to know an answer came from the open web."""

    class StubWeb:
        async def search(self, query: str, *, max_results: int = 5) -> list[RetrievedChunk]:
            return [chunk("web:1", source=RetrievalSource.WEB)]

    retriever = HybridRetriever(
        retrievers=[StubRetriever("dense", RetrievalSource.DENSE, [chunk("a")])],
        config=HybridConfig(use_corrective=True, use_rerank=False),
        evaluator=RetrievalEvaluator(router=make_router(["0"])),
        web_search=StubWeb(),
    )

    result = await retriever.retrieve(RetrievalRequest(query="q"), tenant_id=TENANT)

    assert result.web_fallback_used is True
    assert any(c.source is RetrievalSource.WEB for c in result.chunks)


async def test_disabled_web_search_returns_nothing_rather_than_guessing() -> None:
    """Saying "I could not find it" beats answering from unrelated context."""
    assert await NoWebSearch().search("q", max_results=5) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("8", 0.8), ("Score: 10/10", 1.0), ("0", 0.0), ("I cannot grade this", 0.5), ("42", 1.0)],
)
def test_grade_parsing_tolerates_chatty_models(raw: str, expected: float) -> None:
    """An unparseable grade is not evidence of irrelevance."""
    assert parse_score(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.9, CragVerdict.CORRECT), (0.5, CragVerdict.AMBIGUOUS), (0.1, CragVerdict.INCORRECT)],
)
def test_verdict_thresholds(score: float, expected: CragVerdict) -> None:
    """Route boundaries are asymmetric on purpose and must stay that way."""
    assert verdict_from_score(score, thresholds=CragThresholds()) is expected


# ── Query rewriting ──────────────────────────────────────────────────────────


async def test_hyde_expansion_is_added_to_the_search() -> None:
    """The hypothetical answer is embedded, never shown."""
    rewriter = QueryRewriter(router=make_router(["Incremental compilation is disabled when..."]))

    result = await rewriter.rewrite("why is my build slow?", use_hyde=True)

    assert result.hypothetical_document is not None
    assert result.hypothetical_document in result.expansions
    assert "hyde" in result.techniques


async def test_multi_query_generates_distinct_variants() -> None:
    """Fusing several phrasings beats guessing the one right phrasing."""
    router = make_router(
        ["build performance issues\nslow compilation causes\nwhy builds take long"]
    )
    rewriter = QueryRewriter(router=router)

    result = await rewriter.rewrite("why is my build slow?", use_multi_query=True, variants=3)

    assert len(result.expansions) == 3
    assert "multi_query" in result.techniques


async def test_rewrite_failure_falls_back_to_the_original_query() -> None:
    """A rewrite is an optimisation; losing it must not lose the answer."""
    pricing.MODEL_PROVIDERS["rewrite-dead-model"] = "fake"
    router = LLMRouter(
        providers={"fake": FakeProvider(fail_times=99)},
        policy=ModelPolicy(default_model="rewrite-dead-model"),
    )

    result = await QueryRewriter(router=router).rewrite("q", use_hyde=True, use_multi_query=True)

    assert result.expansions == ()
    assert result.all_queries == ("q",)
    assert result.errors


async def test_decomposition_reports_sub_questions() -> None:
    """Multi-hop questions need separate retrievals, not one blended one."""
    router = make_router(["What is EMEA revenue?\nWhat is APAC revenue?"])

    result = await QueryRewriter(router=router).rewrite("compare EMEA and APAC", decompose=True)

    assert result.is_multi_hop is True
    assert len(result.sub_questions) == 2


def test_parse_lines_strips_numbering_and_bullets() -> None:
    """Models add formatting no matter how firmly the prompt forbids it."""
    raw = chr(10).join(["1. first query", "- second query", '"third query"'])

    assert _parse_lines(raw, limit=5, exclude=None) == [
        "first query",
        "second query",
        "third query",
    ]


def test_parse_lines_drops_a_variant_identical_to_the_original() -> None:
    """A duplicate variant would double the original's weight in fusion."""
    assert _parse_lines("only one", limit=5, exclude="only one") == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("what is our latest pricing?", True),
        ("has this changed recently?", True),
        ("what does RAG stand for?", False),
    ],
)
def test_time_sensitivity_heuristic(query: str, expected: bool) -> None:
    """Cheap enough to run before deciding whether to pay for a classifier."""
    assert looks_time_sensitive(query) is expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [("compare our EMEA and APAC revenue", True), ("what is our EMEA revenue?", False)],
)
def test_multi_hop_heuristic(query: str, expected: bool) -> None:
    """Pre-filter so decomposition is only attempted when plausibly useful."""
    assert looks_multi_hop(query) is expected


# ── Reranking ────────────────────────────────────────────────────────────────


async def test_identity_reranker_preserves_order() -> None:
    """The measurement baseline must not reorder anything."""
    hits = [chunk("a"), chunk("b"), chunk("c")]

    reranked = await IdentityReranker().rerank("q", hits, top_n=2)

    assert [c.chunk_id for c in reranked] == ["a", "b"]


async def test_reranked_chunks_carry_their_score() -> None:
    """The score feeds the citation binder and the drift dashboard."""
    reranked = await ScriptedReranker({"a": 0.42}).rerank("q", [chunk("a")], top_n=1)

    assert reranked[0].rerank_score == pytest.approx(0.42)
    assert reranked[0].effective_score == pytest.approx(0.42)


async def test_reranking_an_empty_list_is_a_no_op() -> None:
    """An empty retrieval must not cost a model call."""
    assert await ScriptedReranker({}).rerank("q", [], top_n=5) == []


# ── Backend details ──────────────────────────────────────────────────────────


def test_sparse_index_names_are_per_tenant_and_sanitised() -> None:
    """Per-tenant indices mean a naming bug finds nothing rather than leaking."""
    assert index_name_for("ten_ABC") == "agrag-chunks-ten_abc"
    assert index_name_for("ten/../other") == "agrag-chunks-ten----other"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("has acquired", "HAS_ACQUIRED"), ("reports-to", "REPORTS_TO"), ("!!!", "RELATED_TO")],
)
def test_relation_normalisation(raw: str, expected: str) -> None:
    """Unnormalised labels make traversal silently miss most of the graph."""
    assert normalise_relation(raw) == expected


def test_heuristic_entity_extraction_ignores_sentence_case() -> None:
    """The leading capital of a sentence is grammar, not a name."""
    assert heuristic_entities("Did ACME acquire Globex?") == ["ACME", "Globex"]
    assert heuristic_entities("what is our revenue?") == []


# ── In-memory retriever ──────────────────────────────────────────────────────


async def test_in_memory_retriever_scopes_to_its_tenant() -> None:
    """Even the test double must not serve another tenant's corpus."""
    retriever = InMemoryRetriever([chunk("a", content="alpha")], tenant_id="ten_a")

    assert await retriever.retrieve(RetrievalRequest(query="alpha"), tenant_id="ten_b") == []


async def test_in_memory_retriever_ranks_by_term_overlap() -> None:
    """Predictable ranking is the point; quality is not."""
    corpus = [
        chunk("a", content="retrieval augmented generation"),
        chunk("b", content="unrelated text"),
    ]

    hits = await InMemoryRetriever(corpus).retrieve(
        RetrievalRequest(query="retrieval generation"), tenant_id="ten_test"
    )

    assert [h.chunk_id for h in hits] == ["a"]


# ── Configuration ────────────────────────────────────────────────────────────


def test_strategies_map_onto_configuration() -> None:
    """A tenant's enabled_strategies column drives the whole pipeline."""
    config = HybridConfig.from_strategies(["hybrid", "hyde", "corrective", "adaptive"])

    assert (config.use_dense, config.use_sparse) == (True, True)
    assert (config.use_hyde, config.use_corrective, config.use_adaptive) == (True, True, True)
    assert config.use_graph is False


def test_unknown_strategy_names_are_ignored_not_fatal() -> None:
    """A tenant row from a newer deployment must not break an older worker."""
    config = HybridConfig.from_strategies(["hybrid", "quantum_retrieval_9000"])

    assert config.use_dense is True


def test_dense_only_tenant_does_not_enable_sparse() -> None:
    """Explicit configuration must be honoured exactly."""
    assert HybridConfig.from_strategies(["dense"]).use_sparse is False
