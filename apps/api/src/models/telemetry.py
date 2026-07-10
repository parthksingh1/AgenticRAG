"""Operational records: guardrails, retrieval, cost, audit and webhooks.

These tables back the admin dashboards. They are append-only and deliberately
denormalised — a dashboard query should never need to join five tables to draw
one line chart, and retention is managed by partition-style pruning jobs rather
than cascading deletes.

Example:
    >>> from src.models.telemetry import GuardrailEvent, GuardrailDecision
    >>> GuardrailDecision.BLOCK.value
    'block'
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, JSONColumn, TenantScoped, TimestampMixin, new_id


class GuardrailStage(StrEnum):
    """Whether a guardrail ran on the way in or on the way out."""

    INPUT = "input"
    OUTPUT = "output"


class GuardrailKind(StrEnum):
    """Which guardrail produced the event."""

    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    PII = "pii"
    TOXICITY = "toxicity"
    OFF_TOPIC = "off_topic"
    MODERATION = "moderation"
    HALLUCINATION = "hallucination"
    CITATION = "citation"
    COST_CAP = "cost_cap"
    RATE_LIMIT = "rate_limit"


class GuardrailDecision(StrEnum):
    """What the guardrail did about it."""

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"
    FLAG = "flag"


class GuardrailEvent(Base, TenantScoped, TimestampMixin):
    """One guardrail decision. Mirrored to an OTel span attribute at runtime."""

    __tablename__ = "guardrail_events"
    __table_args__ = (
        Index("ix_guardrail_events_tenant_id_kind_created_at", "tenant_id", "kind", "created_at"),
        Index("ix_guardrail_events_decision", "decision"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("grd"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    message_id: Mapped[str | None] = mapped_column(String(64), index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(64))
    user_id: Mapped[str | None] = mapped_column(String(64))
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True)

    stage: Mapped[GuardrailStage] = mapped_column(String(10), nullable=False)
    kind: Mapped[GuardrailKind] = mapped_column(String(30), nullable=False)
    decision: Mapped[GuardrailDecision] = mapped_column(String(10), nullable=False)
    detector: Mapped[str | None] = mapped_column(
        String(60), comment="Which detector fired: heuristic, classifier, llm_judge, nli."
    )
    score: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[str | None] = mapped_column(Text)
    # Never store the offending text verbatim for PII events — only the entity
    # types found, so the audit trail is not itself a PII store.
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)


class RetrievalLog(Base, TenantScoped, TimestampMixin):
    """Per-retrieval-call record, used for drift detection and failure triage."""

    __tablename__ = "retrieval_logs"
    __table_args__ = (
        Index("ix_retrieval_logs_tenant_id_created_at", "tenant_id", "created_at"),
        Index("ix_retrieval_logs_strategy", "strategy"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("ret"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    message_id: Mapped[str | None] = mapped_column(String(64), index=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True)

    query: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_queries: Mapped[list[str]] = mapped_column(JSONColumn, default=list, nullable=False)
    strategy: Mapped[str] = mapped_column(String(40), nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    expanded: Mapped[bool] = mapped_column(
        Boolean, default=False, comment="True when adaptive retrieval widened k."
    )

    result_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_ids: Mapped[list[str]] = mapped_column(JSONColumn, default=list, nullable=False)
    scores: Mapped[list[str]] = mapped_column(
        JSONColumn,
        default=list,
        nullable=False,
        comment="Post-fusion scores, aligned with chunk_ids.",
    )
    rerank_scores: Mapped[list[str]] = mapped_column(JSONColumn, default=list, nullable=False)
    top_score: Mapped[float | None] = mapped_column(Float, index=True)
    mean_score: Mapped[float | None] = mapped_column(Float)

    crag_verdict: Mapped[str | None] = mapped_column(
        String(20), comment="correct | ambiguous | incorrect, from the retrieval evaluator."
    )
    web_fallback_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    dense_latency_ms: Mapped[int | None] = mapped_column(Integer)
    sparse_latency_ms: Mapped[int | None] = mapped_column(Integer)
    rerank_latency_ms: Mapped[int | None] = mapped_column(Integer)
    total_latency_ms: Mapped[int | None] = mapped_column(Integer)


class UsageRecord(Base, TenantScoped, TimestampMixin):
    """A single billable LLM or embedding call.

    One row per provider call rather than per request, so the cost dashboard can
    attribute spend to the specific graph node (rewriter, judge, generator) that
    incurred it.
    """

    __tablename__ = "usage_records"
    __table_args__ = (
        Index("ix_usage_records_tenant_id_usage_date", "tenant_id", "usage_date"),
        Index("ix_usage_records_tenant_id_model", "tenant_id", "model"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("use"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64))
    conversation_id: Mapped[str | None] = mapped_column(String(64))
    message_id: Mapped[str | None] = mapped_column(String(64), index=True)
    trace_id: Mapped[str | None] = mapped_column(String(64))

    usage_date: Mapped[datetime] = mapped_column(Date, nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    operation: Mapped[str] = mapped_column(
        String(40), nullable=False, comment="chat | embedding | rerank | judge | classify."
    )
    node: Mapped[str | None] = mapped_column(String(60), comment="LangGraph node that called it.")

    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    was_fallback: Mapped[bool] = mapped_column(
        Boolean, default=False, comment="True when the primary provider failed over."
    )


class TenantBudgetCounter(Base, TenantScoped, TimestampMixin):
    """Durable daily token/cost counters.

    Redis holds the hot counter for rate limiting; this table is the source of
    truth that survives a Redis flush and backs the billing view.
    """

    __tablename__ = "tenant_budget_counters"
    __table_args__ = (
        UniqueConstraint("tenant_id", "usage_date", name="uq_tenant_budget_counters_tenant_date"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("bdg"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    usage_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    tokens_used: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0, nullable=False)
    requests: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class AuditLog(Base, TenantScoped, TimestampMixin):
    """Append-only record of every administrative action.

    Written inside the same transaction as the action itself, so an action that
    rolls back leaves no audit entry and vice versa.
    """

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_tenant_id_created_at", "tenant_id", "created_at"),
        Index("ix_audit_logs_actor_user_id", "actor_user_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("aud"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    actor_user_id: Mapped[str | None] = mapped_column(String(64))
    actor_api_key_id: Mapped[str | None] = mapped_column(String(64))
    actor_ip: Mapped[str | None] = mapped_column(String(45))

    action: Mapped[str] = mapped_column(
        String(80), nullable=False, comment="e.g. tenant.config.update, api_key.revoke."
    )
    resource_type: Mapped[str | None] = mapped_column(String(60))
    resource_id: Mapped[str | None] = mapped_column(String(64))
    before: Mapped[dict[str, Any] | None] = mapped_column(JSONColumn)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSONColumn)
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(64))


class WebhookEndpoint(Base, TenantScoped, TimestampMixin):
    """A tenant-registered HTTP endpoint for platform events."""

    __tablename__ = "webhook_endpoints"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("whk"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    events: Mapped[list[str]] = mapped_column(
        JSONColumn,
        default=list,
        nullable=False,
        comment="Subscribed event names, e.g. ingestion.completed, cost.threshold_hit.",
    )
    secret_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="HMAC key hash; signatures use the plaintext."
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String(300))
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class WebhookDelivery(Base, TenantScoped, TimestampMixin):
    """One delivery attempt, retained so tenants can debug their integrations."""

    __tablename__ = "webhook_deliveries"
    __table_args__ = (Index("ix_webhook_deliveries_endpoint_created", "endpoint_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("whd"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    endpoint_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event: Mapped[str] = mapped_column(String(60), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status_code: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
