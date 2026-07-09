"""Eval, judge-calibration, prompt-registry and experiment models.

These tables make the evals story auditable: every number that appears in the
README or on a dashboard traces back to a row here, written by
``python -m evals.run``. Nothing is hand-entered.

Example:
    >>> from src.models.evaluation import EvalSetName
    >>> EvalSetName.ADVERSARIAL.value
    'adversarial'
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, JSONColumn, TenantScoped, TimestampMixin, new_id


class EvalSetName(StrEnum):
    """The three standing eval sets, plus ad-hoc canary runs."""

    GOLDEN = "golden"
    REGRESSION = "regression"
    ADVERSARIAL = "adversarial"
    CANARY = "canary"


class EvalRunStatus(StrEnum):
    """Lifecycle of an eval run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvalRun(Base, TimestampMixin):
    """One invocation of the eval harness over one set.

    Deliberately *not* tenant-scoped: eval runs are a platform-level artefact
    executed against the fixture tenant, and CI needs to read them without a
    tenant context.
    """

    __tablename__ = "eval_runs"
    __table_args__ = (Index("ix_eval_runs_set_name_created_at", "set_name", "created_at"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("evr"))
    set_name: Mapped[EvalSetName] = mapped_column(String(20), nullable=False)
    set_version: Mapped[str] = mapped_column(String(20), default="v1", nullable=False)
    status: Mapped[EvalRunStatus] = mapped_column(
        String(20), default=EvalRunStatus.RUNNING, nullable=False
    )

    # Exactly what produced these numbers, so a result is reproducible.
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str | None] = mapped_column(String(50))
    retrieval_strategies: Mapped[list[str]] = mapped_column(
        JSONColumn, default=list, nullable=False
    )
    git_sha: Mapped[str | None] = mapped_column(String(40), index=True)
    git_branch: Mapped[str | None] = mapped_column(String(200))
    pr_number: Mapped[int | None] = mapped_column(Integer, index=True)
    triggered_by: Mapped[str] = mapped_column(
        String(40), default="manual", nullable=False, comment="manual | ci | nightly | canary."
    )

    case_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    passed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Ragas + custom metrics, e.g. {"faithfulness": 0.91, "citation_precision": 0.88}
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)
    baseline_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONColumn)
    metric_deltas: Mapped[dict[str, Any] | None] = mapped_column(JSONColumn)
    gate_passed: Mapped[bool | None] = mapped_column(
        Boolean, comment="Null until the CI gate has evaluated this run."
    )
    gate_failures: Mapped[list[str]] = mapped_column(JSONColumn, default=list, nullable=False)

    total_cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0, nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    report_url: Mapped[str | None] = mapped_column(Text)
    langfuse_url: Mapped[str | None] = mapped_column(Text)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    results: Mapped[list[EvalCaseResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvalCaseResult(Base, TimestampMixin):
    """The outcome of one eval case within a run."""

    __tablename__ = "eval_case_results"
    __table_args__ = (
        Index("ix_eval_case_results_run_id_passed", "run_id", "passed"),
        UniqueConstraint("run_id", "case_id", name="uq_eval_case_results_run_id_case_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("evc"))
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False
    )
    case_id: Mapped[str] = mapped_column(String(80), nullable=False)
    intent: Mapped[str | None] = mapped_column(String(40), index=True)
    difficulty: Mapped[str | None] = mapped_column(String(20))

    query: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(Text)
    actual_answer: Mapped[str | None] = mapped_column(Text)
    expected_sources: Mapped[list[str]] = mapped_column(JSONColumn, default=list, nullable=False)
    retrieved_sources: Mapped[list[str]] = mapped_column(JSONColumn, default=list, nullable=False)

    metrics: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)
    judge_scores: Mapped[dict[str, Any]] = mapped_column(
        JSONColumn,
        default=dict,
        nullable=False,
        comment="Raw per-judge scores before calibration, keyed by judge model.",
    )
    calibrated_score: Mapped[float | None] = mapped_column(Float)
    judges_disagreed: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )
    human_label: Mapped[float | None] = mapped_column(
        Float, comment="Set by an admin in the disagreement review UI."
    )

    passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    failure_mode: Mapped[str | None] = mapped_column(String(60))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(64))
    langfuse_url: Mapped[str | None] = mapped_column(Text)

    run: Mapped[EvalRun] = relationship(back_populates="results")


class JudgeCalibration(Base, TimestampMixin):
    """Weekly calibration coefficients for one LLM judge.

    The final score for a case is the inverse-ECE-weighted average of the judges,
    so a judge that is systematically overconfident contributes less. Recomputed
    by ``evals/scripts/calibrate_judges.py`` against a 50-case human-labelled
    subset.
    """

    __tablename__ = "judge_calibrations"
    __table_args__ = (
        UniqueConstraint("judge_model", "computed_at", name="uq_judge_calibrations_model_time"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("jcal"))
    judge_model: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    calibration_set_version: Mapped[str] = mapped_column(String(20), default="v1", nullable=False)

    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_calibration_error: Mapped[float] = mapped_column(Float, nullable=False)
    cohens_kappa: Mapped[float | None] = mapped_column(
        Float, comment="Agreement with the other judge on the same subset."
    )
    pearson_r: Mapped[float | None] = mapped_column(Float)
    mean_absolute_error: Mapped[float | None] = mapped_column(Float)

    # Platt-style affine correction applied to the raw judge score.
    slope: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    intercept: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    weight: Mapped[float] = mapped_column(
        Float, default=0.5, nullable=False, comment="Normalised inverse-ECE blending weight."
    )

    reliability_bins: Mapped[dict[str, Any]] = mapped_column(
        JSONColumn,
        default=dict,
        nullable=False,
        comment="Bin centres and accuracies for the ECE plot.",
    )
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)

    def calibrate(self, raw_score: float) -> float:
        """Apply the affine correction to a raw judge score, clamped to [0, 1].

        Example:
            >>> cal = JudgeCalibration(slope=0.8, intercept=0.1)
            >>> round(cal.calibrate(0.9), 3)
            0.82
        """
        return min(1.0, max(0.0, self.slope * raw_score + self.intercept))


class PromptRecord(Base, TimestampMixin):
    """Registry entry for a versioned prompt loaded from ``prompts/``.

    The YAML files on disk are the source of truth; this table records which
    content hash was seen when, and which version is currently promoted, so the
    playground can diff and the A/B router can address a variant by id.
    """

    __tablename__ = "prompt_records"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_prompt_records_name_version"),
        Index("ix_prompt_records_name_is_active", "name", "is_active"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("prm"))
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    template: Mapped[str] = mapped_column(Text, nullable=False)
    variables: Mapped[list[str]] = mapped_column(JSONColumn, default=list, nullable=False)
    author: Mapped[str | None] = mapped_column(String(200))
    changelog: Mapped[str | None] = mapped_column(Text)

    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    promoted_by: Mapped[str | None] = mapped_column(String(200))
    promoted_from_run_id: Mapped[str | None] = mapped_column(
        String(64), comment="Eval run that justified the promotion."
    )
    langfuse_prompt_id: Mapped[str | None] = mapped_column(String(120))


class ExperimentStatus(StrEnum):
    """Lifecycle of an A/B or canary experiment."""

    DRAFT = "draft"
    RUNNING = "running"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    STOPPED = "stopped"


class Experiment(Base, TenantScoped, TimestampMixin):
    """A live traffic split between a control and a variant.

    Covers both prompt A/B tests and model canaries: ``control`` and ``variant``
    are opaque config blobs interpreted by the router.
    """

    __tablename__ = "experiments"
    __table_args__ = (Index("ix_experiments_tenant_id_status", "tenant_id", "status"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("exp"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    hypothesis: Mapped[str | None] = mapped_column(Text)
    status: Mapped[ExperimentStatus] = mapped_column(
        String(20), default=ExperimentStatus.DRAFT, nullable=False
    )

    control: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)
    variant: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)
    variant_traffic_pct: Mapped[float] = mapped_column(Float, default=5.0, nullable=False)
    is_shadow: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Shadow runs mirror traffic to the variant without returning it.",
    )

    primary_metric: Mapped[str] = mapped_column(String(60), default="faithfulness", nullable=False)
    minimum_sample_size: Mapped[int] = mapped_column(Integer, default=200, nullable=False)
    auto_promote: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    auto_rollback_threshold: Mapped[float] = mapped_column(
        Float, default=0.03, nullable=False, comment="Relative regression that triggers rollback."
    )

    control_stats: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)
    variant_stats: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)
    p_value: Mapped[float | None] = mapped_column(Float)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decision_note: Mapped[str | None] = mapped_column(Text)


class DriftSnapshot(Base, TenantScoped, TimestampMixin):
    """Daily distribution snapshot used for retrieval and embedding drift alerts.

    Stores the histogram rather than raw scores so the comparison window can be
    arbitrarily long without unbounded storage.
    """

    __tablename__ = "drift_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "snapshot_date", "metric", name="uq_drift_snapshots_tenant_date_metric"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("drf"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    snapshot_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metric: Mapped[str] = mapped_column(
        String(60), nullable=False, comment="retrieval_top_score | embedding_centroid | query_len."
    )
    histogram: Mapped[dict[str, Any]] = mapped_column(JSONColumn, default=dict, nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    psi: Mapped[float | None] = mapped_column(
        Float, comment="Population Stability Index vs baseline."
    )
    kl_divergence: Mapped[float | None] = mapped_column(Float)
    alerted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
