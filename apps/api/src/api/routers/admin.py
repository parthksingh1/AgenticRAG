"""Admin surfaces.

Everything the ``/admin`` dashboards read: workspace configuration, API keys and
users, cost, evals, judge calibration, drift, the failure explorer, the prompt
registry, experiments, webhooks, the audit log and the GDPR controls.

Two rules hold across every route here.

**Admin scope is required, and admin is still tenant-scoped.** An admin
administers *their* workspace. The session-level tenant guard applies to these
routes exactly as it does to the chat routes; the only cross-tenant path in the
system is the GDPR cascade, which is deliberately explicit about it.

**Anything that changes state is audited** with the before and after values, so
a configuration change three weeks ago is answerable rather than remembered.

Example:
    >>> from src.api.routers.admin import router
    >>> router.prefix
    '/api/admin'
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query

from src.api.auth import Principal, get_principal
from src.core.errors import ConflictError, NotFoundError, ValidationFailedError
from src.core.logging import get_logger
from src.governance import audit
from src.models.tenant import ApiKeyScope
from src.schemas.admin import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyOut,
    AuditEntryOut,
    CostSummary,
    DriftPoint,
    EvalCaseOut,
    EvalRunOut,
    ExperimentOut,
    FailureCaseOut,
    JudgeCalibrationOut,
    TenantConfigOut,
    TenantConfigUpdate,
    TriageRequest,
    UsagePoint,
    WebhookCreate,
    WebhookOut,
)

log = get_logger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

#: Every route here requires admin scope. Declared once on the router rather
#: than repeated per route, because a scope check that has to be remembered on
#: each new endpoint is a scope check that will eventually be forgotten.
ADMIN = Annotated[Principal, Depends(get_principal)]


def _require_admin(principal: Principal) -> None:
    """Reject a non-admin principal.

    Raises:
        AuthorizationError: when the caller lacks admin scope.
    """
    principal.require(ApiKeyScope.ADMIN)


# ── workspace configuration ───────────────────────────────────────────────────


@router.get("/config", response_model=TenantConfigOut)
async def get_config(principal: ADMIN) -> TenantConfigOut:
    """Read the workspace's configuration."""
    _require_admin(principal)

    from src.core.db import session_scope
    from src.models.tenant import Tenant

    async with session_scope() as session:
        tenant = await session.get(Tenant, principal.tenant_id)
        if tenant is None:
            raise NotFoundError("Workspace not found.")
        return _config_of(tenant)


@router.patch("/config", response_model=TenantConfigOut)
async def update_config(
    update: TenantConfigUpdate,
    principal: ADMIN,
) -> TenantConfigOut:
    """Update the workspace's configuration.

    Changing the embedding model does *not* re-embed the corpus. Existing
    vectors live in the old model's space and comparing across spaces produces
    confident nonsense, so the response says a reindex is required and the
    caller must trigger it deliberately.
    """
    _require_admin(principal)

    from src.core.db import session_scope
    from src.models.tenant import Tenant

    fields = update.model_dump(exclude_unset=True)
    if not fields:
        raise ValidationFailedError("No fields to update.")

    async with session_scope() as session:
        tenant = await session.get(Tenant, principal.tenant_id)
        if tenant is None:
            raise NotFoundError("Workspace not found.")

        before = _config_of(tenant).model_dump(mode="json")
        reindex_needed = (
            "embedding_model" in fields and fields["embedding_model"] != tenant.embedding_model
        )

        for key, value in fields.items():
            if key == "enabled_strategies" and value is not None:
                tenant.enabled_strategies = [str(s) for s in value]
            else:
                setattr(tenant, key, value)

        await session.flush()
        after = _config_of(tenant)

    await audit.record(
        action="tenant.config.updated",
        tenant_id=principal.tenant_id,
        resource_type="tenant",
        resource_id=principal.tenant_id,
        before=before,
        after=after.model_dump(mode="json"),
    )

    if reindex_needed:
        log.warning(
            "the embedding model changed; existing vectors are in a different space and "
            "the corpus must be reindexed before retrieval is trustworthy",
            tenant_id=principal.tenant_id,
        )

    # The tenant runtime is rebuilt from the tenant row on every request, so the
    # change is live on the next one with nothing to invalidate.
    return after


# ── API keys ──────────────────────────────────────────────────────────────────


@router.get("/api-keys", response_model=list[ApiKeyOut])
async def list_api_keys(principal: ADMIN) -> list[ApiKeyOut]:
    """List the workspace's API keys. Secrets are never returned."""
    _require_admin(principal)

    from sqlalchemy import select

    from src.core.db import session_scope
    from src.models.tenant import ApiKey

    async with session_scope() as session:
        keys = (
            (await session.execute(select(ApiKey).order_by(ApiKey.created_at.desc())))
            .scalars()
            .all()
        )
    return [_key_out(key) for key in keys]


@router.post("/api-keys", response_model=ApiKeyCreated, status_code=201)
async def create_api_key(request: ApiKeyCreate, principal: ADMIN) -> ApiKeyCreated:
    """Mint an API key.

    The secret is returned once and stored only as a hash. A key that can be
    re-read from the database is a key that leaks with the database.
    """
    _require_admin(principal)

    from src.api.auth import generate_api_key
    from src.core.db import session_scope
    from src.models.tenant import ApiKey

    secret, prefix, key_hash = generate_api_key()
    expires_at = (
        datetime.now(UTC) + timedelta(days=request.expires_in_days)
        if request.expires_in_days
        else None
    )

    async with session_scope() as session:
        key = ApiKey(
            tenant_id=principal.tenant_id,
            name=request.name,
            key_hash=key_hash,
            prefix=prefix,
            scopes=[str(scope) for scope in request.scopes],
            created_by_user_id=principal.user_id,
            expires_at=expires_at,
        )
        session.add(key)
        await session.flush()
        out = _key_out(key)

    await audit.record(
        action="api_key.created",
        tenant_id=principal.tenant_id,
        resource_type="api_key",
        resource_id=out.id,
        after={"name": request.name, "scopes": list(out.scopes), "prefix": prefix},
    )
    return ApiKeyCreated(**out.model_dump(), secret=secret)


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(key_id: str, principal: ADMIN) -> None:
    """Revoke an API key.

    Revoked rather than deleted: the audit log references the key id, and a
    dangling reference makes "which key did this" unanswerable.
    """
    _require_admin(principal)

    from src.core.db import session_scope
    from src.models.tenant import ApiKey

    async with session_scope() as session:
        key = await session.get(ApiKey, key_id)
        if key is None:
            raise NotFoundError("API key not found.")
        if key.revoked_at is not None:
            raise ConflictError("That key is already revoked.")
        key.revoked_at = datetime.now(UTC)
        name = key.name

    await audit.record(
        action="api_key.revoked",
        tenant_id=principal.tenant_id,
        resource_type="api_key",
        resource_id=key_id,
        before={"name": name, "revoked": False},
        after={"name": name, "revoked": True},
    )


# ── users ─────────────────────────────────────────────────────────────────────


@router.get("/users")
async def list_users(principal: ADMIN) -> list[dict[str, Any]]:
    """List the workspace's members."""
    _require_admin(principal)

    from sqlalchemy import select

    from src.core.db import session_scope
    from src.models.tenant import User

    async with session_scope() as session:
        users = (
            (
                await session.execute(
                    select(User).where(User.deleted_at.is_(None)).order_by(User.created_at)
                )
            )
            .scalars()
            .all()
        )

    return [
        {
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "role": user.role,
            "last_seen_at": user.last_seen_at.isoformat() if user.last_seen_at else None,
            "created_at": user.created_at.isoformat(),
        }
        for user in users
    ]


# ── cost ──────────────────────────────────────────────────────────────────────


@router.get("/cost", response_model=CostSummary)
async def cost_summary(
    principal: ADMIN,
    days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> CostSummary:
    """Spend, token usage and anomalies for the cost dashboard."""
    _require_admin(principal)

    from src.core.db import session_scope
    from src.models.tenant import Tenant
    from src.repositories.usage import budget_status_for, cost_anomalies, usage_by_day

    async with session_scope() as session:
        rows = await usage_by_day(session, tenant_id=principal.tenant_id, days=days)
        anomalies = await cost_anomalies(session, tenant_id=principal.tenant_id, days=days)
        tenant = await session.get(Tenant, principal.tenant_id)
        budget = await budget_status_for(
            session,
            tenant_id=principal.tenant_id,
            tokens_limit=tenant.daily_token_budget if tenant else 0,
        )

    by_model: dict[str, float] = {}
    total_cost = total_tokens = 0.0
    requests = 0
    points: list[UsagePoint] = []

    for row in rows:
        cost = float(row["cost_usd"] or 0)
        prompt_tokens = int(row["prompt_tokens"] or 0)
        completion_tokens = int(row["completion_tokens"] or 0)

        by_model[row["model"]] = round(by_model.get(row["model"], 0.0) + cost, 6)
        total_cost += cost
        total_tokens += prompt_tokens + completion_tokens
        requests += int(row["requests"] or 0)
        points.append(
            UsagePoint(
                usage_date=row["usage_date"],
                model=row["model"],
                provider=row["provider"],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=round(cost, 6),
                requests=int(row["requests"] or 0),
            )
        )

    limit = int(budget["tokens_limit"] or 0)
    return CostSummary(
        total_cost_usd=round(total_cost, 6),
        total_tokens=int(total_tokens),
        requests=requests,
        by_model=by_model,
        by_day=tuple(points),
        budget_remaining_tokens=int(budget["tokens_remaining"]),
        budget_fraction_used=round(int(budget["tokens_used"]) / limit, 4) if limit else 0.0,
        anomalies=tuple(anomalies),
    )


# ── evals ─────────────────────────────────────────────────────────────────────


@router.get("/evals/runs", response_model=list[EvalRunOut])
async def list_eval_runs(
    principal: ADMIN,
    set_name: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> list[EvalRunOut]:
    """List eval runs, newest first.

    Eval runs are not tenant-scoped: they measure the system, not a workspace,
    and the golden set is the same for everyone. Admin scope still gates access.
    """
    _require_admin(principal)

    from sqlalchemy import select

    from src.core.db import system_session
    from src.models.evaluation import EvalRun

    async with system_session(reason="admin eval dashboard") as session:
        statement = select(EvalRun).order_by(EvalRun.created_at.desc()).limit(limit)
        if set_name:
            statement = statement.where(EvalRun.set_name == set_name)
        runs = (await session.execute(statement)).scalars().all()

    return [_run_out(run) for run in runs]


@router.get("/evals/runs/{run_id}", response_model=EvalRunOut)
async def get_eval_run(run_id: str, principal: ADMIN) -> EvalRunOut:
    """Load one eval run."""
    _require_admin(principal)

    from src.core.db import system_session
    from src.models.evaluation import EvalRun

    async with system_session(reason="admin eval dashboard") as session:
        run = await session.get(EvalRun, run_id)
        if run is None:
            raise NotFoundError("Eval run not found.")
        return _run_out(run)


@router.get("/evals/runs/{run_id}/cases", response_model=list[EvalCaseOut])
async def list_eval_cases(
    run_id: str,
    principal: ADMIN,
    failed_only: Annotated[bool, Query()] = False,
    disagreed_only: Annotated[bool, Query()] = False,
) -> list[EvalCaseOut]:
    """List a run's cases.

    ``disagreed_only`` is the human-in-the-loop queue: the cases where the two
    judges disagreed are exactly the ones worth a human label, because they are
    where the automated score is least trustworthy.
    """
    _require_admin(principal)

    from sqlalchemy import select

    from src.core.db import system_session
    from src.models.evaluation import EvalCaseResult

    async with system_session(reason="admin eval dashboard") as session:
        statement = select(EvalCaseResult).where(EvalCaseResult.run_id == run_id)
        if failed_only:
            statement = statement.where(EvalCaseResult.passed.is_(False))
        if disagreed_only:
            statement = statement.where(EvalCaseResult.judges_disagreed.is_(True))
        cases = (await session.execute(statement.order_by(EvalCaseResult.case_id))).scalars().all()

    return [
        EvalCaseOut(
            case_id=case.case_id,
            intent=case.intent,
            difficulty=case.difficulty,
            query=case.query,
            expected_answer=case.expected_answer,
            actual_answer=case.actual_answer,
            expected_sources=tuple(case.expected_sources or ()),
            retrieved_sources=tuple(case.retrieved_sources or ()),
            metrics=dict(case.metrics or {}),
            judge_scores=dict(case.judge_scores or {}),
            calibrated_score=case.calibrated_score,
            judges_disagreed=case.judges_disagreed,
            human_label=case.human_label,
            passed=case.passed,
            failure_mode=case.failure_mode,
            langfuse_url=case.langfuse_url,
        )
        for case in cases
    ]


@router.post("/evals/cases/{case_id}/label", status_code=204)
async def label_eval_case(
    case_id: str,
    principal: ADMIN,
    label: Annotated[float, Body(embed=True, ge=0.0, le=1.0)],
) -> None:
    """Attach a human label to an eval case.

    These labels are the calibration set. Every one added makes the next weekly
    recalibration better, which is why the disagreement queue exists at all.
    """
    _require_admin(principal)

    from src.core.db import system_session
    from src.models.evaluation import EvalCaseResult

    async with system_session(reason="human eval labelling") as session:
        case = await session.get(EvalCaseResult, case_id)
        if case is None:
            raise NotFoundError("Eval case not found.")
        case.human_label = label


@router.get("/evals/calibration", response_model=list[JudgeCalibrationOut])
async def judge_calibration(principal: ADMIN) -> list[JudgeCalibrationOut]:
    """Current judge calibrations, for the calibration viewer."""
    _require_admin(principal)

    from sqlalchemy import select

    from src.core.db import system_session
    from src.models.evaluation import JudgeCalibration

    async with system_session(reason="admin calibration viewer") as session:
        rows = (
            (
                await session.execute(
                    select(JudgeCalibration)
                    .where(JudgeCalibration.is_active.is_(True))
                    .order_by(JudgeCalibration.judge_model)
                )
            )
            .scalars()
            .all()
        )

    return [
        JudgeCalibrationOut(
            judge_model=row.judge_model,
            sample_size=row.sample_size,
            expected_calibration_error=row.expected_calibration_error,
            cohens_kappa=row.cohens_kappa,
            pearson_r=row.pearson_r,
            slope=row.slope,
            intercept=row.intercept,
            weight=row.weight,
            reliability_bins=dict(row.reliability_bins or {}),
            computed_at=row.computed_at,
        )
        for row in rows
    ]


# ── drift ─────────────────────────────────────────────────────────────────────


@router.get("/drift", response_model=list[DriftPoint])
async def drift_series(
    principal: ADMIN,
    metric: Annotated[str, Query()] = "retrieval_top_score",
    days: Annotated[int, Query(ge=1, le=365)] = 60,
) -> list[DriftPoint]:
    """A metric's drift history, oldest first for charting."""
    _require_admin(principal)

    from sqlalchemy import select

    from src.core.db import session_scope
    from src.governance.drift import severity
    from src.models.evaluation import DriftSnapshot

    since = datetime.now(UTC) - timedelta(days=days)
    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(DriftSnapshot)
                    .where(DriftSnapshot.metric == metric, DriftSnapshot.snapshot_date >= since)
                    .order_by(DriftSnapshot.snapshot_date)
                )
            )
            .scalars()
            .all()
        )

    return [
        DriftPoint(
            snapshot_date=row.snapshot_date.date(),
            metric=row.metric,
            psi=row.psi,
            severity=severity(row.psi) if row.psi is not None else "unknown",
            sample_size=row.sample_size,
            kl_divergence=row.kl_divergence,
            histogram=dict(row.histogram or {}),
            alerted=row.alerted,
        )
        for row in rows
    ]


# ── failure explorer ──────────────────────────────────────────────────────────


@router.get("/failures", response_model=list[FailureCaseOut])
async def list_failures(
    principal: ADMIN,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    untriaged_only: Annotated[bool, Query()] = True,
) -> list[FailureCaseOut]:
    """Thumbs-down feedback, newest first.

    This is where the regression set grows from. A failure a user reported is
    worth more as a test case than any synthetic query, because it is a question
    someone actually asked and did not get answered.
    """
    _require_admin(principal)

    from sqlalchemy import select

    from src.core.db import session_scope
    from src.models.conversation import FeedbackRating, Message, MessageFeedback

    async with session_scope() as session:
        statement = (
            select(MessageFeedback, Message)
            .join(Message, Message.id == MessageFeedback.message_id)
            .where(MessageFeedback.rating == FeedbackRating.DOWN)
            .order_by(MessageFeedback.created_at.desc())
            .limit(limit)
        )
        if untriaged_only:
            statement = statement.where(MessageFeedback.failure_mode.is_(None))
        rows = (await session.execute(statement)).all()

        out = []
        for feedback, message in rows:
            question = (
                await session.execute(
                    select(Message.content)
                    .where(
                        Message.conversation_id == message.conversation_id,
                        Message.created_at <= message.created_at,
                        Message.role == "user",
                    )
                    .order_by(Message.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

            out.append(
                FailureCaseOut(
                    feedback_id=feedback.id,
                    message_id=message.id,
                    conversation_id=message.conversation_id,
                    query=question or "",
                    answer=message.content,
                    comment=feedback.comment,
                    failure_mode=feedback.failure_mode,
                    created_at=feedback.created_at,
                )
            )

    return out


@router.post("/failures/{feedback_id}/triage", status_code=204)
async def triage_failure(feedback_id: str, request: TriageRequest, principal: ADMIN) -> None:
    """Label a failure with its mode, and optionally promote it to a test case."""
    _require_admin(principal)

    from src.core.db import session_scope
    from src.models.conversation import MessageFeedback

    async with session_scope() as session:
        feedback = await session.get(MessageFeedback, feedback_id)
        if feedback is None:
            raise NotFoundError("Feedback not found.")
        feedback.failure_mode = request.failure_mode
        feedback.triaged_by_user_id = principal.user_id
        feedback.triaged_at = datetime.now(UTC)

        if request.promote_to_regression_set:
            if not request.expected_answer:
                msg = (
                    "promoting to the regression set requires an expected answer; "
                    "a case with nothing to grade against always passes"
                )
                raise ValidationFailedError(msg)
            feedback.promoted_to_regression_set = True

    log.info(
        "triaged a failure",
        feedback_id=feedback_id,
        failure_mode=request.failure_mode,
        promoted=request.promote_to_regression_set,
    )


# ── prompt registry ───────────────────────────────────────────────────────────


@router.get("/prompts")
async def list_prompts(principal: ADMIN) -> list[dict[str, Any]]:
    """List every prompt version, with its content hash and active flag.

    The content hash is what makes a prompt version verifiable: two deployments
    claiming the same version can be checked rather than trusted.
    """
    _require_admin(principal)

    from src.services.prompts import get_prompt_registry

    registry = get_prompt_registry()
    out = []
    for name in registry.names():
        active = registry.active_version(name)
        for version in registry.versions_of(name):
            prompt = registry.get(name, version)
            out.append(
                {
                    "name": name,
                    "version": version,
                    "content_hash": prompt.content_hash,
                    "is_active": version == active,
                    "variables": sorted(prompt.declared_variables()),
                    "changelog": prompt.changelog,
                    "author": prompt.author,
                }
            )
    return out


@router.post("/prompts/{name}/promote/{version}", status_code=204)
async def promote_prompt(name: str, version: str, principal: ADMIN) -> None:
    """Promote a prompt version to active for this process.

    In-process only, and deliberately so: the YAML files under ``prompts/`` are
    the source of truth, and a promotion that lives only in one replica's memory
    is a playground affordance, not a deployment. Making it stick means changing
    the file and shipping it, which is the point — a prompt change that skips
    review skips the eval gate with it.
    """
    _require_admin(principal)

    from src.services.prompts import get_prompt_registry

    registry = get_prompt_registry()
    try:
        registry.promote(name, version)
    except KeyError as exc:
        raise NotFoundError(f"No prompt {name}@{version}.") from exc

    await audit.record(
        action="prompt.promoted",
        tenant_id=principal.tenant_id,
        resource_type="prompt",
        resource_id=f"{name}@{version}",
        after={"name": name, "version": version, "scope": "process-local"},
    )


# ── experiments ───────────────────────────────────────────────────────────────


@router.get("/experiments", response_model=list[ExperimentOut])
async def list_experiments(principal: ADMIN) -> list[ExperimentOut]:
    """List A/B experiments."""
    _require_admin(principal)

    from sqlalchemy import select

    from src.core.db import session_scope
    from src.models.evaluation import Experiment

    async with session_scope() as session:
        rows = (
            (await session.execute(select(Experiment).order_by(Experiment.created_at.desc())))
            .scalars()
            .all()
        )

    return [
        ExperimentOut(
            id=row.id,
            name=row.name,
            hypothesis=row.hypothesis,
            status=row.status,
            control=dict(row.control or {}),
            variant=dict(row.variant or {}),
            variant_traffic_pct=row.variant_traffic_pct,
            is_shadow=row.is_shadow,
            primary_metric=row.primary_metric,
            control_stats=dict(row.control_stats or {}),
            variant_stats=dict(row.variant_stats or {}),
            p_value=row.p_value,
            # Not stored: derived from p_value at read time using the
            # conventional 0.05 threshold, so a p_value edited by hand can never
            # disagree with the significance flag shown next to it.
            is_significant=(row.p_value < 0.05) if row.p_value is not None else None,
            decided_at=row.decided_at,
            created_at=row.created_at,
        )
        for row in rows
    ]


# ── webhooks ──────────────────────────────────────────────────────────────────


@router.get("/webhooks", response_model=list[WebhookOut])
async def list_webhooks(principal: ADMIN) -> list[WebhookOut]:
    """List the workspace's webhook endpoints."""
    _require_admin(principal)

    from sqlalchemy import select

    from src.core.db import session_scope
    from src.models.telemetry import WebhookEndpoint

    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(WebhookEndpoint).order_by(WebhookEndpoint.created_at.desc())
                )
            )
            .scalars()
            .all()
        )

    return [
        WebhookOut(
            id=row.id,
            url=row.url,
            events=tuple(row.events or ()),
            is_active=row.is_active,
            consecutive_failures=row.consecutive_failures,
            disabled_at=row.disabled_at,
            created_at=row.created_at,
        )
        for row in rows
    ]


@router.post("/webhooks", response_model=dict, status_code=201)
async def create_webhook(request: WebhookCreate, principal: ADMIN) -> dict[str, Any]:
    """Register a webhook endpoint.

    The URL is validated against the SSRF guard *at registration* as well as at
    delivery, so a bad endpoint is rejected with a clear message rather than
    silently failing every delivery afterwards.

    The signing secret is returned once.
    """
    _require_admin(principal)

    from src.core.db import session_scope
    from src.core.net import validate_public_url
    from src.models.telemetry import WebhookEndpoint
    from src.services.webhooks import generate_secret, hash_secret

    validate_public_url(str(request.url))
    secret = generate_secret()

    async with session_scope() as session:
        endpoint = WebhookEndpoint(
            tenant_id=principal.tenant_id,
            url=str(request.url),
            events=[str(event) for event in request.events],
            secret_hash=hash_secret(secret),
            description=request.description,
        )
        session.add(endpoint)
        await session.flush()
        endpoint_id = endpoint.id

    await audit.record(
        action="webhook.created",
        tenant_id=principal.tenant_id,
        resource_type="webhook",
        resource_id=endpoint_id,
        after={"url": str(request.url), "events": list(request.events)},
    )
    return {
        "id": endpoint_id,
        "url": str(request.url),
        "events": list(request.events),
        "secret": secret,
        "note": "Store this secret now. It cannot be retrieved again.",
    }


@router.delete("/webhooks/{endpoint_id}", status_code=204)
async def delete_webhook(endpoint_id: str, principal: ADMIN) -> None:
    """Delete a webhook endpoint."""
    _require_admin(principal)

    from sqlalchemy import delete

    from src.core.db import session_scope
    from src.models.telemetry import WebhookEndpoint

    async with session_scope() as session:
        endpoint = await session.get(WebhookEndpoint, endpoint_id)
        if endpoint is None:
            raise NotFoundError("Webhook endpoint not found.")
        url = endpoint.url
        await session.execute(delete(WebhookEndpoint).where(WebhookEndpoint.id == endpoint_id))

    await audit.record(
        action="webhook.deleted",
        tenant_id=principal.tenant_id,
        resource_type="webhook",
        resource_id=endpoint_id,
        before={"url": url},
    )


# ── audit log ─────────────────────────────────────────────────────────────────


@router.get("/audit", response_model=list[AuditEntryOut])
async def list_audit_entries(
    principal: ADMIN,
    action: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AuditEntryOut]:
    """Read the workspace's audit log, newest first."""
    _require_admin(principal)

    from sqlalchemy import select

    from src.core.db import session_scope
    from src.models.telemetry import AuditLog

    async with session_scope() as session:
        statement = (
            select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)
        )
        if action:
            statement = statement.where(AuditLog.action == action)
        rows = (await session.execute(statement)).scalars().all()

    return [
        AuditEntryOut(
            id=row.id,
            action=row.action,
            actor_user_id=row.actor_user_id,
            actor_api_key_id=row.actor_api_key_id,
            actor_ip=row.actor_ip,
            resource_type=row.resource_type,
            resource_id=row.resource_id,
            before=row.before,
            after=row.after,
            success=row.success,
            created_at=row.created_at,
        )
        for row in rows
    ]


# ── governance ────────────────────────────────────────────────────────────────


@router.get("/export")
async def export_workspace(principal: ADMIN) -> dict[str, Any]:
    """Export everything held about this workspace (GDPR portability)."""
    _require_admin(principal)

    from src.governance.gdpr import export_tenant

    return await export_tenant(principal.tenant_id)


@router.post("/erase")
async def erase_workspace(
    principal: ADMIN,
    dry_run: Annotated[bool, Query()] = True,
    confirm: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    """Erase this workspace across every store (GDPR erasure).

    Defaults to a dry run, and a real erasure requires ``confirm`` to equal the
    tenant id. This is the only operation in the system with no undo, and a
    misplaced click should not be able to trigger it.

    Raises:
        ValidationFailedError: when a real erasure is requested without the
            matching confirmation.
    """
    _require_admin(principal)

    from src.governance.gdpr import erase_tenant

    if not dry_run and confirm != principal.tenant_id:
        msg = "a real erasure requires confirm to equal the workspace id"
        raise ValidationFailedError(msg)

    return await erase_tenant(principal.tenant_id, dry_run=dry_run)


# ── helpers ───────────────────────────────────────────────────────────────────


def _config_of(tenant: Any) -> TenantConfigOut:
    """Map a tenant row onto its API shape."""
    return TenantConfigOut(
        id=tenant.id,
        name=tenant.name,
        slug=tenant.slug,
        plan=tenant.plan,
        embedding_model=tenant.embedding_model,
        chunking_strategy=tenant.chunking_strategy,
        enabled_strategies=tuple(tenant.enabled_strategies or ()),
        contextual_retrieval_enabled=tenant.contextual_retrieval_enabled,
        graph_extraction_enabled=tenant.graph_extraction_enabled,
        custom_instructions=tenant.custom_instructions,
        response_template=tenant.response_template,
        daily_token_budget=tenant.daily_token_budget,
        monthly_cost_cap_usd=float(tenant.monthly_cost_cap_usd)
        if tenant.monthly_cost_cap_usd is not None
        else None,
        data_region=tenant.data_region,
        retention_days=tenant.retention_days,
    )


def _key_out(key: Any) -> ApiKeyOut:
    """Map an API key row onto its API shape, without the secret."""
    return ApiKeyOut(
        id=key.id,
        name=key.name,
        prefix=key.prefix,
        scopes=tuple(key.scopes or ()),
        created_at=key.created_at,
        last_used_at=key.last_used_at,
        expires_at=key.expires_at,
        revoked_at=key.revoked_at,
    )


def _run_out(run: Any) -> EvalRunOut:
    """Map an eval run onto its API shape."""
    return EvalRunOut(
        id=run.id,
        set_name=run.set_name,
        set_version=run.set_version,
        status=run.status,
        model=run.model,
        prompt_version=run.prompt_version,
        git_sha=run.git_sha,
        pr_number=run.pr_number,
        triggered_by=run.triggered_by,
        case_count=run.case_count,
        passed_count=run.passed_count,
        failed_count=run.failed_count,
        metrics={k: float(v) for k, v in (run.metrics or {}).items() if _numeric(v)},
        metric_deltas={k: float(v) for k, v in (run.metric_deltas or {}).items() if _numeric(v)}
        if run.metric_deltas
        else None,
        gate_passed=run.gate_passed,
        gate_failures=tuple(run.gate_failures or ()),
        total_cost_usd=float(run.total_cost_usd or 0),
        duration_seconds=run.duration_seconds,
        created_at=run.created_at,
        report_url=run.report_url,
    )


def _numeric(value: Any) -> bool:
    """Whether a stored metric value is a number.

    Metrics dictionaries also carry per-metric notes and sample counts; feeding
    those to ``float`` would raise and take the whole dashboard down over a
    string.

    Example:
        >>> _numeric(0.5), _numeric("n/a"), _numeric(None)
        (True, False, False)
    """
    return isinstance(value, int | float) and not isinstance(value, bool)
