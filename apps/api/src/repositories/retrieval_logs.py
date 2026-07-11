"""Retrieval telemetry.

Every retrieval is logged with its scores, latency and strategy. That is what
makes the drift dashboard possible — a score distribution shifting over time is
the earliest signal that an embedding model, a corpus or a user population has
changed — and what lets a failure in the explorer be traced to what was actually
retrieved rather than guessed at.

Writes are best effort: telemetry that can fail a user's search has the priority
backwards.
"""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)

#: Chunk ids and scores kept per log row. The full result set is unbounded and
#: the tail is not what anyone inspects.
MAX_LOGGED_RESULTS = 20


async def record_retrieval_log(*, tenant_id: str, request: Any, result: Any) -> None:
    """Persist one retrieval for drift analysis and failure triage."""
    from src.core.db import session_scope
    from src.models.telemetry import RetrievalLog
    from src.observability.tracing import current_trace_id

    chunks = list(result.chunks[:MAX_LOGGED_RESULTS])

    async with session_scope() as session:
        session.add(
            RetrievalLog(
                tenant_id=tenant_id,
                trace_id=current_trace_id(),
                query=request.query[:4000],
                rewritten_queries=list(getattr(request, "expansions", ()) or ()),
                strategy=result.strategy,
                top_k=request.top_k,
                expanded=result.expanded,
                result_count=len(result.chunks),
                chunk_ids=[c.chunk_id for c in chunks],
                scores=[round(c.effective_score, 6) for c in chunks],
                rerank_scores=[
                    round(c.rerank_score, 6) for c in chunks if c.rerank_score is not None
                ],
                top_score=result.top_score,
                mean_score=result.mean_score,
                crag_verdict=result.crag_verdict.value if result.crag_verdict else None,
                web_fallback_used=result.web_fallback_used,
                total_latency_ms=result.latency_ms,
                dense_latency_ms=result.source_latencies_ms.get("pgvector"),
                sparse_latency_ms=result.source_latencies_ms.get("opensearch-bm25"),
            )
        )
