"""Drift detection.

Retrieval quality degrades quietly. Nobody files a ticket saying "the corpus has
shifted"; they say the answers got worse three weeks ago and cannot say when. A
daily histogram of retrieval scores, compared against a baseline window, turns
that into a dated signal.

The metric is the **Population Stability Index**, the standard measure for
comparing two distributions of the same variable over time:

    PSI = sum over bins of (actual_i - expected_i) * ln(actual_i / expected_i)

Read with the conventional thresholds: below 0.1 is stable, 0.1 to 0.25 is a
moderate shift worth watching, above 0.25 is a material shift worth
investigating. Those cutoffs are heuristics from credit-risk modelling, not laws
— they are documented here so a reader knows the alert is a prompt to look, not
a verdict.

Storing histograms rather than raw scores keeps the comparison window unbounded
without unbounded storage, which matters because the interesting comparison is
often against a month ago, not yesterday.

Example:
    >>> psi([0.5, 0.5], [0.5, 0.5])
    0.0
    >>> psi([0.9, 0.1], [0.1, 0.9]) > 0.25
    True
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)

#: Histogram bins for scores in [0, 1].
BINS = 10

#: Conventional PSI interpretation thresholds.
PSI_MODERATE = 0.1
PSI_MATERIAL = 0.25

#: Days of history a snapshot is compared against.
BASELINE_WINDOW_DAYS = 14

#: Below this many samples a day's histogram is too sparse to compare, and
#: comparing it anyway produces alerts driven by a quiet weekend.
MIN_SAMPLES = 50

#: Added to empty bins so the logarithm stays finite. Without it a single bin
#: that is empty on one side sends PSI to infinity and every comparison alerts.
EPSILON = 1e-6

METRICS = ("retrieval_top_score", "retrieval_mean_score", "query_length", "result_count")


def histogram(values: Sequence[float], *, bins: int = BINS, upper: float = 1.0) -> list[float]:
    """Bin values into normalised proportions.

    Args:
        values: The raw observations.
        bins: How many equal-width bins to use.
        upper: The top of the range. Values above it land in the last bin rather
            than being dropped, because a score that has moved out of range is
            exactly the signal this is looking for.

    Example:
        >>> histogram([0.05, 0.95], bins=2)
        [0.5, 0.5]
        >>> histogram([], bins=2)
        [0.0, 0.0]
    """
    counts = [0] * bins
    if not values:
        return [0.0] * bins

    width = upper / bins
    for value in values:
        index = min(bins - 1, max(0, int(value / width)) if width else 0)
        counts[index] += 1

    total = len(values)
    return [round(count / total, 6) for count in counts]


def psi(actual: Sequence[float], expected: Sequence[float]) -> float:
    """Population Stability Index between two normalised histograms.

    Raises:
        ValueError: when the histograms have different bin counts. Comparing
            them would silently truncate one and report a fabricated number.

    Example:
        >>> round(psi([0.6, 0.4], [0.5, 0.5]), 4)
        0.0405
    """
    if len(actual) != len(expected):
        msg = f"histograms must have the same bin count; got {len(actual)} and {len(expected)}"
        raise ValueError(msg)

    total = 0.0
    for a, e in zip(actual, expected, strict=True):
        a_safe = max(a, EPSILON)
        e_safe = max(e, EPSILON)
        total += (a_safe - e_safe) * math.log(a_safe / e_safe)
    return round(total, 6)


def severity(value: float) -> str:
    """Label a PSI value.

    Example:
        >>> severity(0.05), severity(0.15), severity(0.4)
        ('stable', 'moderate', 'material')
    """
    if value >= PSI_MATERIAL:
        return "material"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


async def snapshot_tenant(tenant_id: str, *, day: datetime | None = None) -> dict[str, Any]:
    """Snapshot one tenant's retrieval distributions for a day.

    Compares each metric against the mean of the previous
    :data:`BASELINE_WINDOW_DAYS` snapshots — a single previous day is far too
    noisy to alert on, and one bad Tuesday would page someone every week.
    """
    from sqlalchemy import select

    from src.core.context import request_context
    from src.core.db import session_scope
    from src.models.evaluation import DriftSnapshot
    from src.models.telemetry import RetrievalLog

    when = (day or datetime.now(UTC)).replace(hour=0, minute=0, second=0, microsecond=0)
    since = when - timedelta(days=1)

    results: dict[str, Any] = {}
    with request_context(tenant_id=tenant_id):
        async with session_scope() as session:
            logs = (
                (
                    await session.execute(
                        select(RetrievalLog).where(
                            RetrievalLog.created_at >= since, RetrievalLog.created_at < when
                        )
                    )
                )
                .scalars()
                .all()
            )

            if len(logs) < MIN_SAMPLES:
                log.debug(
                    "not enough retrievals to snapshot drift",
                    tenant_id=tenant_id,
                    samples=len(logs),
                )
                return {"tenant_id": tenant_id, "skipped": "insufficient samples"}

            for metric in METRICS:
                values, upper = _series(logs, metric)
                bins = histogram(values, upper=upper)

                baseline = await _baseline(session, metric=metric, before=when)
                score = psi(bins, baseline) if baseline else None
                level = severity(score) if score is not None else "unknown"

                session.add(
                    DriftSnapshot(
                        tenant_id=tenant_id,
                        snapshot_date=when,
                        metric=metric,
                        histogram={"bins": bins, "upper": upper},
                        sample_size=len(values),
                        psi=score,
                        alerted=level == "material",
                    )
                )
                results[metric] = {"psi": score, "severity": level, "samples": len(values)}

                if level == "material":
                    log.warning(
                        "material distribution shift detected",
                        tenant_id=tenant_id,
                        metric=metric,
                        psi=score,
                    )

    return {"tenant_id": tenant_id, "date": when.date().isoformat(), "metrics": results}


async def snapshot_all_tenants() -> dict[str, Any]:
    """Snapshot every active tenant. Called nightly by the beat schedule."""
    from sqlalchemy import select

    from src.core.db import system_session
    from src.models.tenant import Tenant

    async with system_session(reason="nightly drift snapshot") as session:
        tenant_ids = [
            row[0]
            for row in (
                await session.execute(select(Tenant.id).where(Tenant.deleted_at.is_(None)))
            ).all()
        ]

    snapshots = []
    for tenant_id in tenant_ids:
        try:
            snapshots.append(await snapshot_tenant(tenant_id))
        except Exception as exc:  # noqa: BLE001 - one tenant must not stop the sweep
            log.error("drift snapshot failed", tenant_id=tenant_id, error=str(exc))

    return {"tenants": len(tenant_ids), "snapshots": snapshots}


async def _baseline(session: Any, *, metric: str, before: datetime) -> list[float]:
    """Average the previous window's histograms into one baseline.

    Returns an empty list when there is no history, which the caller reads as
    "no comparison possible" rather than "no drift".
    """
    from sqlalchemy import select

    from src.models.evaluation import DriftSnapshot

    rows = (
        (
            await session.execute(
                select(DriftSnapshot)
                .where(
                    DriftSnapshot.metric == metric,
                    DriftSnapshot.snapshot_date < before,
                    DriftSnapshot.snapshot_date >= before - timedelta(days=BASELINE_WINDOW_DAYS),
                )
                .order_by(DriftSnapshot.snapshot_date.desc())
            )
        )
        .scalars()
        .all()
    )
    histograms = [row.histogram.get("bins", []) for row in rows if row.histogram]
    histograms = [h for h in histograms if len(h) == BINS]
    if not histograms:
        return []

    return [round(sum(column) / len(histograms), 6) for column in zip(*histograms, strict=True)]


def _series(logs: Sequence[Any], metric: str) -> tuple[list[float], float]:
    """Extract one metric's values and its expected upper bound.

    The bound differs per metric — scores are in [0, 1], query lengths are not —
    and binning a 400-character query against a [0, 1] range would put every
    observation in the last bin and report perfect stability forever.

    Example:
        >>> _series([], "query_length")
        ([], 500.0)
    """
    if metric == "retrieval_top_score":
        return ([float(row.top_score) for row in logs if row.top_score is not None], 1.0)
    if metric == "retrieval_mean_score":
        return ([float(row.mean_score) for row in logs if row.mean_score is not None], 1.0)
    if metric == "query_length":
        return ([float(len(row.query or "")) for row in logs], 500.0)
    return ([float(row.result_count) for row in logs], 50.0)
