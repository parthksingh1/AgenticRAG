"""Repositories: all database access lives behind these functions.

Routes and services never write queries. That separation is what makes the
tenant isolation tests meaningful — there is a finite, reviewable set of places
that touch the database — and it is what lets the query for "conversations in
the sidebar" be optimised once rather than in three routes that each grew their
own version.

Every function here takes an explicit session. Opening one internally would hide
the transaction boundary, and a service that needs two writes to be atomic could
not express it.

Example:
    >>> from src.repositories.usage import today
    >>> len(today()) == 10
    True
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger
from src.models.telemetry import TenantBudgetCounter, UsageRecord

log = get_logger(__name__)


def today() -> str:
    """Today's date in UTC as an ISO string.

    UTC rather than local time so a tenant's daily budget resets at the same
    instant everywhere, and two workers in different regions agree on which day
    a request belongs to.

    Example:
        >>> today() == datetime.now(UTC).date().isoformat()
        True
    """
    return datetime.now(UTC).date().isoformat()


async def record_usage(*, tenant_id: str, completion: Any, request: Any) -> None:
    """Persist one provider call and roll it into the day's counter.

    Both writes happen in one transaction: a usage row without its counter
    increment would make the budget under-count, and a counter increment without
    its row would make the cost dashboard unable to explain the spend.
    """
    from src.core.db import session_scope

    async with session_scope() as session:
        session.add(
            UsageRecord(
                tenant_id=tenant_id,
                usage_date=date.fromisoformat(today()),
                provider=completion.provider,
                model=completion.model,
                operation=_operation_for(request.node),
                node=request.node,
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
                cached_tokens=completion.usage.cached_tokens,
                cost_usd=completion.cost_usd,
                latency_ms=completion.latency_ms,
                was_fallback=completion.was_fallback,
            )
        )
        await _increment_counter(
            session,
            tenant_id=tenant_id,
            tokens=completion.usage.total_tokens,
            cost_usd=completion.cost_usd,
        )


async def _increment_counter(
    session: AsyncSession, *, tenant_id: str, tokens: int, cost_usd: float
) -> None:
    """Add spend to today's durable counter, creating the row if needed."""
    from sqlalchemy.dialects.postgresql import insert

    statement = (
        insert(TenantBudgetCounter)
        .values(
            tenant_id=tenant_id,
            usage_date=date.fromisoformat(today()),
            tokens_used=tokens,
            cost_usd=cost_usd,
            requests=1,
        )
        # An upsert rather than read-modify-write: two workers finishing a turn
        # in the same millisecond would otherwise both read the old value and one
        # increment would be lost.
        .on_conflict_do_update(
            constraint="uq_tenant_budget_counters_tenant_date",
            set_={
                "tokens_used": TenantBudgetCounter.tokens_used + tokens,
                "cost_usd": TenantBudgetCounter.cost_usd + cost_usd,
                "requests": TenantBudgetCounter.requests + 1,
            },
        )
    )
    await session.execute(statement)


def _operation_for(node: str | None) -> str:
    """Map a graph node onto a billing operation.

    Example:
        >>> _operation_for("generator")
        'chat'
        >>> _operation_for("injection_judge")
        'judge'
        >>> _operation_for(None)
        'chat'
    """
    if not node:
        return "chat"
    if "judge" in node or "critic" in node or "evaluator" in node:
        return "judge"
    if "rewriter" in node or "router" in node or "classif" in node:
        return "classify"
    if "embed" in node:
        return "embedding"
    if "rerank" in node:
        return "rerank"
    return "chat"


async def usage_by_day(
    session: AsyncSession, *, tenant_id: str, days: int = 30
) -> list[dict[str, Any]]:
    """Daily spend broken down by model, for the cost dashboard.

    Aggregated in SQL rather than in Python: a busy tenant's month is hundreds of
    thousands of rows, and shipping them all into the API process to sum them is
    both slow and a memory risk.
    """
    from datetime import timedelta

    cutoff = datetime.now(UTC).date() - timedelta(days=days)
    result = await session.execute(
        select(
            UsageRecord.usage_date,
            UsageRecord.model,
            UsageRecord.provider,
            func.sum(UsageRecord.prompt_tokens).label("prompt_tokens"),
            func.sum(UsageRecord.completion_tokens).label("completion_tokens"),
            func.sum(UsageRecord.cost_usd).label("cost_usd"),
            func.count().label("requests"),
        )
        .where(UsageRecord.tenant_id == tenant_id, UsageRecord.usage_date >= cutoff)
        .group_by(UsageRecord.usage_date, UsageRecord.model, UsageRecord.provider)
        .order_by(UsageRecord.usage_date.desc())
    )
    return [dict(row) for row in result.mappings().all()]


async def cost_anomalies(
    session: AsyncSession, *, tenant_id: str, days: int = 30, sigma: float = 3.0
) -> list[str]:
    """Days whose spend exceeds the mean by ``sigma`` standard deviations.

    Computed the same way as the Prometheus alert rule, so the dashboard and the
    pager cannot disagree about what counts as an anomaly.

    Fewer than a week of history returns nothing: a standard deviation over three
    points flags almost anything, and a new tenant would be permanently alarmed.
    """
    import statistics

    rows = await usage_by_day(session, tenant_id=tenant_id, days=days)
    by_day: dict[str, float] = {}
    for row in rows:
        key = str(row["usage_date"])
        by_day[key] = by_day.get(key, 0.0) + float(row["cost_usd"] or 0)

    if len(by_day) < 7:
        return []

    values = list(by_day.values())
    mean = statistics.fmean(values)
    deviation = statistics.pstdev(values)
    if deviation == 0:
        return []

    return sorted(day for day, spend in by_day.items() if spend > mean + sigma * deviation)


async def budget_status_for(
    session: AsyncSession, *, tenant_id: str, tokens_limit: int
) -> dict[str, Any]:
    """Today's durable budget counter, used when Redis is unavailable.

    The Redis counter is the hot path; this is the source of truth that survives
    a cache flush, so a flush cannot hand every tenant an unlimited budget.
    """
    result = await session.execute(
        select(TenantBudgetCounter).where(
            TenantBudgetCounter.tenant_id == tenant_id,
            TenantBudgetCounter.usage_date == date.fromisoformat(today()),
        )
    )
    counter = result.scalar_one_or_none()
    used = counter.tokens_used if counter else 0
    return {
        "tokens_used": used,
        "tokens_limit": tokens_limit,
        "tokens_remaining": max(tokens_limit - used, 0),
        "cost_usd": float(counter.cost_usd) if counter else 0.0,
        "requests": counter.requests if counter else 0,
    }
