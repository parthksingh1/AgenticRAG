"""Admin, tenant configuration and public-API schemas."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.models.tenant import ApiKeyScope, ChunkingStrategy, RetrievalStrategy, TenantPlan


class TenantConfigOut(BaseModel):
    """A workspace's current configuration."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    slug: str
    plan: TenantPlan
    embedding_model: str
    chunking_strategy: ChunkingStrategy
    enabled_strategies: tuple[str, ...] = ()
    contextual_retrieval_enabled: bool = True
    graph_extraction_enabled: bool = False
    custom_instructions: str | None = None
    response_template: str | None = None
    daily_token_budget: int
    monthly_cost_cap_usd: float | None = None
    data_region: str
    retention_days: int | None = None


class TenantConfigUpdate(BaseModel):
    """A partial update to workspace configuration.

    Every field is optional and ``None`` means "leave unchanged", so a client
    that sends one field cannot accidentally clear the rest — the failure mode
    of a PUT-shaped update against a config object.
    """

    model_config = ConfigDict(frozen=True)

    name: str | None = Field(default=None, min_length=1, max_length=200)
    embedding_model: str | None = None
    chunking_strategy: ChunkingStrategy | None = None
    enabled_strategies: tuple[RetrievalStrategy, ...] | None = None
    contextual_retrieval_enabled: bool | None = None
    graph_extraction_enabled: bool | None = None
    custom_instructions: str | None = Field(default=None, max_length=8000)
    response_template: str | None = Field(default=None, max_length=8000)
    guardrail_config: dict[str, Any] | None = None
    model_policy: dict[str, Any] | None = None
    daily_token_budget: int | None = Field(default=None, ge=1000)
    monthly_cost_cap_usd: float | None = Field(default=None, ge=0)
    retention_days: int | None = Field(default=None, ge=1)


class ApiKeyOut(BaseModel):
    """An API key, never including the secret."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    prefix: str
    scopes: tuple[str, ...]
    created_at: datetime
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        """Whether the key can still authenticate."""
        return self.revoked_at is None


class ApiKeyCreated(ApiKeyOut):
    """A newly created key, including the secret.

    The only response that ever carries the plaintext. Only the hash is stored,
    so a lost key is regenerated rather than recovered — which is the point.
    """

    secret: str = Field(description="Shown once. It cannot be retrieved again.")


class ApiKeyCreate(BaseModel):
    """Request a new API key."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=200)
    scopes: tuple[ApiKeyScope, ...] = (ApiKeyScope.READ,)
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)


class UsagePoint(BaseModel):
    """One day of usage for the cost dashboard."""

    model_config = ConfigDict(frozen=True)

    usage_date: date
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    requests: int = 0


class CostSummary(BaseModel):
    """Aggregated spend for a period."""

    model_config = ConfigDict(frozen=True)

    total_cost_usd: float = 0.0
    total_tokens: int = 0
    requests: int = 0
    by_model: dict[str, float] = Field(default_factory=dict)
    by_day: tuple[UsagePoint, ...] = ()
    budget_remaining_tokens: int = 0
    budget_fraction_used: float = 0.0
    #: Days whose spend is more than three standard deviations above the
    #: trailing mean. Surfaced rather than merely alerted on, so the dashboard
    #: shows the same anomalies the pager does.
    anomalies: tuple[str, ...] = ()


class EvalRunOut(BaseModel):
    """An eval run, for the evals dashboard."""

    model_config = ConfigDict(frozen=True)

    id: str
    set_name: str
    set_version: str
    status: str
    model: str
    prompt_version: str | None = None
    git_sha: str | None = None
    pr_number: int | None = None
    triggered_by: str
    case_count: int
    passed_count: int
    failed_count: int
    metrics: dict[str, float] = Field(default_factory=dict)
    metric_deltas: dict[str, float] | None = None
    gate_passed: bool | None = None
    gate_failures: tuple[str, ...] = ()
    total_cost_usd: float = 0.0
    duration_seconds: float | None = None
    created_at: datetime
    report_url: str | None = None


class EvalCaseOut(BaseModel):
    """One case's result, for drilling into a failure."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    intent: str | None = None
    difficulty: str | None = None
    query: str
    expected_answer: str | None = None
    actual_answer: str | None = None
    expected_sources: tuple[str, ...] = ()
    retrieved_sources: tuple[str, ...] = ()
    metrics: dict[str, float] = Field(default_factory=dict)
    judge_scores: dict[str, Any] = Field(default_factory=dict)
    calibrated_score: float | None = None
    judges_disagreed: bool = False
    human_label: float | None = None
    passed: bool
    failure_mode: str | None = None
    langfuse_url: str | None = None


class JudgeCalibrationOut(BaseModel):
    """A judge's calibration, for the calibration viewer."""

    model_config = ConfigDict(frozen=True)

    judge_model: str
    sample_size: int
    expected_calibration_error: float
    cohens_kappa: float | None = None
    pearson_r: float | None = None
    slope: float
    intercept: float
    weight: float
    reliability_bins: dict[str, Any] = Field(default_factory=dict)
    computed_at: datetime


class FailureCaseOut(BaseModel):
    """A thumbs-down conversation, for the failure explorer."""

    model_config = ConfigDict(frozen=True)

    feedback_id: str
    message_id: str
    conversation_id: str
    query: str
    answer: str
    comment: str | None = None
    failure_mode: str | None = None
    created_at: datetime
    promoted_to_regression_set: bool = False
    groundedness_score: float | None = None
    guardrail_flags: tuple[str, ...] = ()
    langfuse_url: str | None = None


class TriageRequest(BaseModel):
    """Label a failure and optionally promote it into the regression set."""

    model_config = ConfigDict(frozen=True)

    failure_mode: str = Field(
        min_length=1,
        max_length=60,
        description="e.g. hallucination, missing_citation, wrong_retrieval, refused_wrongly",
    )
    promote_to_regression_set: bool = False
    expected_answer: str | None = Field(
        default=None,
        description="Required to promote: a regression case with no expected answer "
        "cannot be graded, so promoting without one would silently add a case "
        "that always passes.",
    )


class DriftPoint(BaseModel):
    """One day of a distribution, for the drift charts."""

    model_config = ConfigDict(frozen=True)

    snapshot_date: date
    metric: str
    sample_size: int
    psi: float | None = None
    severity: str = Field(
        default="unknown",
        description="stable | moderate | material | unknown, from the PSI thresholds.",
    )
    kl_divergence: float | None = None
    histogram: dict[str, Any] = Field(default_factory=dict)
    alerted: bool = False


class ExperimentOut(BaseModel):
    """A running or finished A/B experiment."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    hypothesis: str | None = None
    status: str
    control: dict[str, Any] = Field(default_factory=dict)
    variant: dict[str, Any] = Field(default_factory=dict)
    variant_traffic_pct: float
    is_shadow: bool
    primary_metric: str
    control_stats: dict[str, Any] = Field(default_factory=dict)
    variant_stats: dict[str, Any] = Field(default_factory=dict)
    p_value: float | None = None
    #: Null until the minimum sample size is reached. Showing a winner before
    #: then is how teams promote noise.
    is_significant: bool | None = None
    decided_at: datetime | None = None


class AuditEntryOut(BaseModel):
    """One administrative action."""

    model_config = ConfigDict(frozen=True)

    id: str
    created_at: datetime
    actor_user_id: str | None = None
    actor_api_key_id: str | None = None
    actor_ip: str | None = None
    action: str
    resource_type: str | None = None
    resource_id: str | None = None
    # The diff is the reason the log is worth reading: "someone changed the
    # guardrail config" is not an answer to an incident, and the before/after
    # pair is. Secrets are redacted before they are ever written.
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    success: bool


class WebhookCreate(BaseModel):
    """Register a webhook endpoint."""

    model_config = ConfigDict(frozen=True)

    url: str = Field(min_length=1, max_length=2000)
    events: tuple[str, ...] = Field(min_length=1)
    description: str | None = Field(default=None, max_length=300)


class WebhookOut(BaseModel):
    """A registered webhook, never including its signing secret."""

    model_config = ConfigDict(frozen=True)

    id: str
    url: str
    events: tuple[str, ...]
    is_active: bool
    created_at: datetime
    consecutive_failures: int = 0
    disabled_at: datetime | None = None


# ── OpenAI-compatible surface ────────────────────────────────────────────────


class OpenAIMessage(BaseModel):
    """A message in the OpenAI chat format."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None


class OpenAIChatRequest(BaseModel):
    """An OpenAI-compatible completion request.

    Accepting this shape means any OpenAI SDK works against this system without
    modification, which is the difference between an API someone can try in two
    minutes and one they have to learn first. Unsupported parameters are
    accepted and ignored rather than rejected, because a client sending
    ``frequency_penalty`` should still get an answer.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    model: str
    messages: tuple[OpenAIMessage, ...] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    user: str | None = None


class OpenAIChoice(BaseModel):
    """One choice in an OpenAI-compatible response."""

    model_config = ConfigDict(frozen=True)

    index: int = 0
    message: OpenAIMessage
    finish_reason: str = "stop"


class OpenAIUsage(BaseModel):
    """Token accounting in the OpenAI shape."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatResponse(BaseModel):
    """An OpenAI-compatible completion response."""

    model_config = ConfigDict(frozen=True)

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: tuple[OpenAIChoice, ...]
    usage: OpenAIUsage
    #: Non-standard, and deliberately so: an OpenAI-shaped RAG response with no
    #: citations would be indistinguishable from a plain completion, which
    #: defeats the point. Clients that ignore unknown fields are unaffected.
    citations: tuple[dict[str, Any], ...] = ()


class BatchRequest(BaseModel):
    """A bulk job for the batch API."""

    model_config = ConfigDict(frozen=True)

    requests: tuple[ChatBatchItem, ...] = Field(min_length=1, max_length=10_000)
    completion_window: Literal["24h"] = "24h"
    metadata: dict[str, str] = Field(default_factory=dict)


class ChatBatchItem(BaseModel):
    """One question in a batch job."""

    model_config = ConfigDict(frozen=True)

    custom_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1)
    model: str | None = None


class BatchOut(BaseModel):
    """A batch job's status."""

    model_config = ConfigDict(frozen=True)

    id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    total: int
    completed: int
    failed: int
    created_at: datetime
    completed_at: datetime | None = None
    output_url: str | None = None
