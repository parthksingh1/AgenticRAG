"""The hybrid retrieval orchestrator.

This is where the strategies compose. Given a tenant's enabled set, it:

1. rewrites the query (HyDE / multi-query / decomposition),
2. fans out to every enabled backend **concurrently**,
3. fuses the ranked lists by reciprocal rank,
4. caps per-document contribution so one long document cannot fill the context,
5. reranks with a cross-encoder,
6. optionally grades the result (CRAG) and, if it is poor, widens k or falls back
   to web search,
7. returns a result carrying the telemetry every one of those steps produced.

Two properties are load-bearing:

**Partial failure degrades, it does not fail the turn.** If OpenSearch is down,
dense results still answer the question. A retrieval stack where one backend
outage takes down chat is worse than one that quietly gets a bit less accurate,
so backend errors are caught per-backend and recorded, not propagated.

**Adaptive widening is bounded.** It happens at most once. An unbounded
"retrieve until confident" loop is how a single ambiguous question turns into
twenty retrievals and a timeout.

Example:
    >>> HybridRetriever.name
    'hybrid'
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.core.logging import get_logger
from src.retrieval.base import Retriever
from src.retrieval.corrective import CragAssessment, RetrievalEvaluator, WebSearchClient
from src.retrieval.fusion import (
    FusionConfig,
    deduplicate_by_document,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from src.retrieval.rerank import Reranker
from src.retrieval.rewrite import (
    QueryRewriter,
    RewriteResult,
    looks_multi_hop,
    looks_time_sensitive,
)
from src.retrieval.types import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalSource,
    RetrievedChunk,
)

log = get_logger(__name__)


@dataclass(slots=True)
class HybridConfig:
    """Which strategies to run and how to combine them.

    Mirrors the tenant's ``enabled_strategies`` column, resolved once per turn so
    the orchestrator never reads the database mid-retrieval.
    """

    use_dense: bool = True
    use_sparse: bool = True
    use_graph: bool = False
    use_colbert: bool = False
    use_hyde: bool = False
    use_multi_query: bool = False
    use_decomposition: bool = False
    use_corrective: bool = False
    use_adaptive: bool = False
    use_temporal: bool = False
    use_rerank: bool = True
    #: Rank fusion is the default; weighted score fusion is the documented
    #: alternative the eval harness compares against.
    fusion_mode: str = "rrf"
    fusion: FusionConfig = field(default_factory=FusionConfig)
    top_k: int = 5
    expanded_k: int = 20
    rerank_top_n: int = 5
    max_per_document: int = 3
    multi_query_variants: int = 3

    @classmethod
    def from_strategies(cls, strategies: Sequence[str], **overrides: object) -> HybridConfig:
        """Build a config from a tenant's enabled strategy names.

        Unknown names are ignored rather than raising: a tenant row written by a
        newer deployment must not break an older worker mid-rollout.

        Example:
            >>> config = HybridConfig.from_strategies(["hybrid", "hyde", "corrective"])
            >>> (config.use_sparse, config.use_hyde, config.use_corrective)
            (True, True, True)
            >>> HybridConfig.from_strategies(["dense"]).use_sparse
            False
        """
        enabled = {s.lower() for s in strategies}
        config = cls(
            use_dense="dense" in enabled or "hybrid" in enabled or not enabled,
            use_sparse="sparse" in enabled or "hybrid" in enabled,
            use_graph="graph" in enabled,
            use_colbert="colbert" in enabled,
            use_hyde="hyde" in enabled,
            use_multi_query="multi_query" in enabled,
            use_decomposition="multi_query" in enabled or "graph" in enabled,
            use_corrective="corrective" in enabled,
            use_adaptive="adaptive" in enabled,
            use_temporal="temporal" in enabled,
        )
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class HybridRetriever:
    """Runs the tenant's enabled retrieval strategies and combines the results."""

    name = "hybrid"

    def __init__(
        self,
        *,
        retrievers: Sequence[Retriever],
        config: HybridConfig | None = None,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
        evaluator: RetrievalEvaluator | None = None,
        web_search: WebSearchClient | None = None,
    ) -> None:
        """Create the orchestrator.

        Args:
            retrievers: Backends to fan out to. Which ones actually run is
                decided by ``config`` and by each backend's source.
            config: Strategy configuration for this tenant.
            reranker: Cross-encoder reranker; skipped when None.
            rewriter: Query rewriter; skipped when None.
            evaluator: CRAG evaluator; skipped when None.
            web_search: Fallback used when CRAG returns ``incorrect``.
        """
        self._retrievers = list(retrievers)
        self._config = config or HybridConfig()
        self._reranker = reranker
        self._rewriter = rewriter
        self._evaluator = evaluator
        self._web_search = web_search

    async def retrieve(self, request: RetrievalRequest, *, tenant_id: str) -> RetrievalResult:
        """Run the full retrieval pipeline for one query."""
        started = time.perf_counter()
        config = self._config

        request = await self._apply_rewriting(request)
        if config.use_temporal and looks_time_sensitive(request.query):
            request = request.model_copy(update={"recency_boost": 0.5})

        chunks, latencies = await self._fan_out(request, tenant_id=tenant_id)
        fused = self._fuse(chunks)
        fused = deduplicate_by_document(fused, max_per_document=config.max_per_document)

        assessment: CragAssessment | None = None
        expanded = False
        web_used = False

        if config.use_corrective and self._evaluator is not None:
            assessment = await self._evaluator.evaluate(request.query, fused[: config.top_k])

            if assessment.needs_wider_retrieval and config.use_adaptive:
                fused, expanded = await self._widen(request, tenant_id=tenant_id, current=fused)
            elif assessment.needs_web_search and self._web_search is not None:
                web_results = await self._web_search.search(request.query, max_results=5)
                if web_results:
                    web_used = True
                    fused = list(assessment.kept) + web_results

        elif config.use_adaptive and _low_confidence(fused, config.top_k):
            fused, expanded = await self._widen(request, tenant_id=tenant_id, current=fused)

        final = await self._rerank(request.query, fused)
        latency_ms = int((time.perf_counter() - started) * 1000)

        return RetrievalResult(
            chunks=tuple(final),
            strategy=self._strategy_label(),
            latency_ms=latency_ms,
            source_latencies_ms=latencies,
            expanded=expanded,
            crag_verdict=assessment.verdict if assessment else None,
            web_fallback_used=web_used,
        )

    # ── stages ───────────────────────────────────────────────────────────────

    async def _apply_rewriting(self, request: RetrievalRequest) -> RetrievalRequest:
        """Attach rewritten query variants to the request."""
        config = self._config
        if self._rewriter is None or not (
            config.use_hyde or config.use_multi_query or config.use_decomposition
        ):
            return request

        rewrite: RewriteResult = await self._rewriter.rewrite(
            request.query,
            use_hyde=config.use_hyde,
            use_multi_query=config.use_multi_query,
            decompose=config.use_decomposition and looks_multi_hop(request.query),
            variants=config.multi_query_variants,
        )
        if not rewrite.expansions:
            return request
        return request.model_copy(
            update={"expansions": tuple(dict.fromkeys(request.expansions + rewrite.expansions))}
        )

    async def _fan_out(
        self, request: RetrievalRequest, *, tenant_id: str
    ) -> tuple[list[list[RetrievedChunk]], dict[str, int]]:
        """Query every enabled backend concurrently, tolerating individual failures."""
        active = [r for r in self._retrievers if self._is_enabled(r)]
        if not active:
            return [], {}

        async def run(retriever: Retriever) -> tuple[str, list[RetrievedChunk], int]:
            start = time.perf_counter()
            try:
                hits = await retriever.retrieve(request, tenant_id=tenant_id)
            except Exception as exc:  # noqa: BLE001 - one backend must not fail the turn
                log.error(
                    "retrieval backend failed; continuing with the others",
                    backend=retriever.name,
                    reason=str(exc),
                )
                hits = []
            return retriever.name, hits, int((time.perf_counter() - start) * 1000)

        results = await asyncio.gather(*(run(r) for r in active))
        latencies = {name: ms for name, _, ms in results}
        return [hits for _, hits, _ in results if hits], latencies

    def _fuse(self, ranked_lists: Sequence[Sequence[RetrievedChunk]]) -> list[RetrievedChunk]:
        """Combine ranked lists using the configured fusion mode."""
        if not ranked_lists:
            return []
        if len(ranked_lists) == 1:
            return list(ranked_lists[0])
        fuse = (
            weighted_score_fusion
            if self._config.fusion_mode == "weighted"
            else (reciprocal_rank_fusion)
        )
        return fuse(ranked_lists, self._config.fusion)

    async def _widen(
        self, request: RetrievalRequest, *, tenant_id: str, current: Sequence[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], bool]:
        """Retry once with a larger k, merging with what was already found.

        Bounded to a single widening on purpose: an unbounded confidence loop is
        how one ambiguous question becomes twenty retrievals and a timeout.
        """
        widened_request = request.model_copy(update={"top_k": self._config.expanded_k})
        wider, _ = await self._fan_out(widened_request, tenant_id=tenant_id)
        if not wider:
            return list(current), False

        merged = self._fuse([*wider, list(current)])
        merged = deduplicate_by_document(merged, max_per_document=self._config.max_per_document)
        log.info("adaptive retrieval widened k", from_k=request.top_k, to_k=self._config.expanded_k)
        return merged, True

    async def _rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Rerank if configured, otherwise truncate to top_k."""
        if not chunks:
            return []
        if self._reranker is None or not self._config.use_rerank:
            return list(chunks[: self._config.top_k])
        return await self._reranker.rerank(query, chunks, top_n=self._config.rerank_top_n)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _is_enabled(self, retriever: Retriever) -> bool:
        """Whether a backend should run under the current configuration."""
        config = self._config
        return {
            RetrievalSource.DENSE: config.use_dense,
            RetrievalSource.SPARSE: config.use_sparse,
            RetrievalSource.GRAPH: config.use_graph,
            RetrievalSource.COLBERT: config.use_colbert,
        }.get(retriever.source, True)

    def _strategy_label(self) -> str:
        """A stable, sorted description of what actually ran.

        Recorded on every retrieval log so a regression can be traced to a
        configuration change rather than guessed at.

        Example:
            >>> HybridRetriever(retrievers=[])._strategy_label()
            'dense+rerank+sparse'
        """
        config = self._config
        parts = [
            name
            for name, on in (
                ("dense", config.use_dense),
                ("sparse", config.use_sparse),
                ("graph", config.use_graph),
                ("colbert", config.use_colbert),
                ("hyde", config.use_hyde),
                ("multi_query", config.use_multi_query),
                ("corrective", config.use_corrective),
                ("adaptive", config.use_adaptive),
                ("temporal", config.use_temporal),
                ("rerank", config.use_rerank),
            )
            if on
        ]
        return "+".join(sorted(parts)) or "none"

    async def aclose(self) -> None:
        """Close every backend and the reranker."""
        await asyncio.gather(*(r.aclose() for r in self._retrievers))
        if self._reranker is not None:
            await self._reranker.aclose()


def _low_confidence(chunks: Sequence[RetrievedChunk], top_k: int) -> bool:
    """Whether a result set looks weak enough to justify widening.

    Two signals: too few results to answer from, or a best score that is barely
    above the rest. Both are cheap proxies used only when CRAG is disabled — with
    CRAG enabled, the evaluator makes the call properly.

    Example:
        >>> _low_confidence([], 5)
        True
    """
    if len(chunks) < max(top_k // 2, 1):
        return True
    scores = [c.effective_score for c in chunks[:top_k]]
    return bool(scores) and max(scores) < 0.35
