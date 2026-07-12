"""Prometheus metrics.

Every metric name here appears in either an alert rule
(``infra/alertmanager/rules``) or a Grafana panel
(``infra/grafana/dashboards``). A metric nobody looks at is a cardinality bill
with no upside, so the set is deliberately small.

Label cardinality is the constraint that shapes the rest. ``tenant_id`` is a
label only where per-tenant breakdown is the point — cost and budget — and never
on latency histograms, where it would multiply every bucket by the tenant count.
Paths are recorded as route templates, so ``/api/documents/{document_id}`` is one
series rather than one per document.

Example:
    >>> record_http_request(method="GET", path="/healthz", status=200, duration_ms=1.2)
"""

from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

#: A private registry rather than the global default, so importing this module
#: twice under a test runner does not raise "duplicate timeseries".
REGISTRY = CollectorRegistry()

#: Latency buckets chosen around the SLOs the alert rules use: the 3s TTFT
#: threshold and the 1.5s retrieval threshold both fall on a bucket edge, so the
#: alert's histogram_quantile does not interpolate across a wide bucket.
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0)
_MS_BUCKETS = tuple(b * 1000 for b in _LATENCY_BUCKETS)

http_requests_total = Counter(
    "http_requests_total",
    "HTTP requests handled.",
    ["method", "path", "status"],
    registry=REGISTRY,
)

http_request_duration_ms = Histogram(
    "http_request_duration_ms",
    "HTTP request duration in milliseconds.",
    ["method", "path"],
    buckets=_MS_BUCKETS,
    registry=REGISTRY,
)

rag_retrieval_latency_ms = Histogram(
    "rag_retrieval_latency_ms",
    "End-to-end retrieval latency in milliseconds.",
    ["strategy"],
    buckets=_MS_BUCKETS,
    registry=REGISTRY,
)

rag_rerank_latency_ms = Histogram(
    "rag_rerank_latency_ms",
    "Reranking latency in milliseconds.",
    ["reranker"],
    buckets=_MS_BUCKETS,
    registry=REGISTRY,
)

rag_ttft_ms = Histogram(
    "rag_ttft_ms",
    "Time to first token in milliseconds.",
    ["model"],
    buckets=_MS_BUCKETS,
    registry=REGISTRY,
)

rag_ttlt_ms = Histogram(
    "rag_ttlt_ms",
    "Time to last token in milliseconds.",
    ["model"],
    buckets=_MS_BUCKETS,
    registry=REGISTRY,
)

rag_tokens_generated = Counter(
    "rag_tokens_generated",
    "Completion tokens generated.",
    ["model"],
    registry=REGISTRY,
)

rag_answers_total = Counter(
    "rag_answers_total",
    "Answers returned to users. The denominator for the hallucination rate.",
    registry=REGISTRY,
)

rag_cache_hits_total = Counter(
    "rag_cache_hits_total",
    "Cache lookups by outcome.",
    ["cache", "outcome"],
    registry=REGISTRY,
)

rag_guardrail_blocks_total = Counter(
    "rag_guardrail_blocks_total",
    "Guardrail decisions that stopped or altered a turn.",
    ["kind", "decision", "stage"],
    registry=REGISTRY,
)

rag_hallucination_detected_total = Counter(
    "rag_hallucination_detected_total",
    "Answers where groundedness checking found unsupported or contradicted claims.",
    ["severity"],
    registry=REGISTRY,
)

rag_cost_usd_total = Counter(
    "rag_cost_usd_total",
    "Cumulative provider spend in USD.",
    ["tenant_id", "model", "operation"],
    registry=REGISTRY,
)

rag_tool_calls_total = Counter(
    "rag_tool_calls_total",
    "MCP tool invocations.",
    ["tool", "outcome"],
    registry=REGISTRY,
)

rag_provider_failures_total = Counter(
    "rag_provider_failures_total",
    "Provider calls that failed, by whether a fallback rescued them.",
    ["provider", "outcome"],
    registry=REGISTRY,
)

ingestion_documents_total = Counter(
    "ingestion_documents_total",
    "Documents processed, by terminal status.",
    ["status"],
    registry=REGISTRY,
)

ingestion_chunks_total = Counter(
    "ingestion_chunks_total",
    "Chunks written to the index.",
    registry=REGISTRY,
)

ingestion_duration_seconds = Histogram(
    "ingestion_duration_seconds",
    "Time to ingest one document end to end.",
    buckets=(1, 5, 15, 30, 60, 120, 300, 600),
    registry=REGISTRY,
)

tenant_budget_used_ratio = Gauge(
    "tenant_budget_used_ratio",
    "Fraction of the daily token budget consumed.",
    ["tenant_id"],
    registry=REGISTRY,
)

active_conversations = Gauge(
    "active_conversations",
    "Conversations with a turn in flight.",
    registry=REGISTRY,
)

backup_runs_total = Counter(
    "backup_runs_total",
    "Backup attempts by component and outcome.",
    ["component", "outcome"],
    registry=REGISTRY,
)

# A gauge rather than a counter: what alerting needs to know is how long it has
# been since the last good backup, which a monotonic counter cannot answer.
backup_last_success_timestamp = Gauge(
    "backup_last_success_timestamp",
    "Unix time of the last verified backup, by component.",
    ["component"],
    registry=REGISTRY,
)

backup_size_bytes = Gauge(
    "backup_size_bytes",
    "Size of the last verified backup, by component.",
    ["component"],
    registry=REGISTRY,
)


def record_http_request(*, method: str, path: str, status: int, duration_ms: float) -> None:
    """Record one HTTP request.

    Example:
        >>> record_http_request(method="GET", path="/api/x", status=200, duration_ms=5.0)
    """
    http_requests_total.labels(method=method, path=path, status=str(status)).inc()
    http_request_duration_ms.labels(method=method, path=path).observe(duration_ms)


def record_retrieval(*, strategy: str, duration_ms: float, rerank_ms: float | None = None) -> None:
    """Record a retrieval and, if it happened, its reranking step."""
    rag_retrieval_latency_ms.labels(strategy=strategy).observe(duration_ms)
    if rerank_ms is not None:
        rag_rerank_latency_ms.labels(reranker="cross-encoder").observe(rerank_ms)


def record_answer(
    *,
    model: str,
    ttft_ms: float | None,
    ttlt_ms: float | None,
    completion_tokens: int,
    groundedness: float | None = None,
) -> None:
    """Record a completed answer and its quality signal.

    ``groundedness`` is bucketed into a severity rather than recorded as a
    histogram: the alert rule asks "what share of answers were ungrounded",
    which needs a counter, and a histogram would not answer it.
    """
    rag_answers_total.inc()
    rag_tokens_generated.labels(model=model).inc(completion_tokens)
    if ttft_ms is not None:
        rag_ttft_ms.labels(model=model).observe(ttft_ms)
    if ttlt_ms is not None:
        rag_ttlt_ms.labels(model=model).observe(ttlt_ms)
    if groundedness is not None and groundedness < 0.8:
        severity = "severe" if groundedness < 0.5 else "minor"
        rag_hallucination_detected_total.labels(severity=severity).inc()


def record_cache(*, cache: str, hit: bool, false_hit: bool = False) -> None:
    """Record a cache lookup.

    ``false_hit`` is its own outcome rather than a miss, because a semantic cache
    that returns the wrong answer and one that returns nothing are very different
    problems and must not aggregate together.
    """
    outcome = "false_hit" if false_hit else ("hit" if hit else "miss")
    rag_cache_hits_total.labels(cache=cache, outcome=outcome).inc()


def record_guardrail(*, kind: str, decision: str, stage: str) -> None:
    """Record a guardrail decision."""
    rag_guardrail_blocks_total.labels(kind=kind, decision=decision, stage=stage).inc()


def record_cost(*, tenant_id: str, model: str, operation: str, cost_usd: float) -> None:
    """Record provider spend."""
    if cost_usd > 0:
        rag_cost_usd_total.labels(tenant_id=tenant_id, model=model, operation=operation).inc(
            cost_usd
        )


def record_tool_call(*, tool: str, ok: bool) -> None:
    """Record an MCP tool invocation."""
    rag_tool_calls_total.labels(tool=tool, outcome="ok" if ok else "error").inc()


def record_provider_failure(*, provider: str, recovered: bool) -> None:
    """Record a provider failure and whether a fallback saved the turn."""
    rag_provider_failures_total.labels(
        provider=provider, outcome="recovered" if recovered else "failed"
    ).inc()


def record_ingestion(
    *, status: str, chunks: int = 0, duration_seconds: float | None = None
) -> None:
    """Record the outcome of one document ingestion."""
    ingestion_documents_total.labels(status=status).inc()
    if chunks:
        ingestion_chunks_total.inc(chunks)
    if duration_seconds is not None:
        ingestion_duration_seconds.observe(duration_seconds)


def record_backup(*, component: str, success: bool, size_bytes: int | None = None) -> None:
    """Record a backup attempt.

    The success timestamp is what the alert rule watches: a backup job that
    stops running produces no failure metric at all, so "no successful backup in
    36 hours" is the only condition that catches both a broken job and a job
    that never started.
    """
    import time

    backup_runs_total.labels(component=component, outcome="success" if success else "failure").inc()
    if success:
        backup_last_success_timestamp.labels(component=component).set(time.time())
        if size_bytes is not None:
            backup_size_bytes.labels(component=component).set(size_bytes)


def set_budget_used(*, tenant_id: str, ratio: float) -> None:
    """Publish a tenant's budget consumption."""
    tenant_budget_used_ratio.labels(tenant_id=tenant_id).set(min(max(ratio, 0.0), 1.0))


def render() -> bytes:
    """Render the registry in Prometheus exposition format.

    Example:
        >>> b"http_requests_total" in render()
        True
    """
    return generate_latest(REGISTRY)


def content_type() -> str:
    """The content type Prometheus expects."""
    from prometheus_client import CONTENT_TYPE_LATEST

    return str(CONTENT_TYPE_LATEST)


def snapshot() -> dict[str, Any]:
    """A small dict of current values, for the admin dashboard's live tiles.

    Reads the registry rather than keeping a parallel set of counters, so the
    dashboard and the alerts can never disagree about what a number is.
    """
    values: dict[str, Any] = {}
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name.endswith(("_total", "_ratio")):
                key = sample.name
                values[key] = values.get(key, 0.0) + sample.value
    return values
