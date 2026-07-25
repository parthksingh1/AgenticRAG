"""Judge calibration.

An LLM judging another LLM's answers is only useful if we know how much to
trust it. A judge that says "0.9" on answers that humans score 0.6 is not
broken — it is *miscalibrated*, and miscalibration is correctable once measured.
Treating an uncalibrated judge's score as ground truth is how eval suites end up
reporting improvements that no user experiences.

This module measures three things against a set of human-labelled cases:

* **Expected calibration error (ECE)** — bin the judge's scores by confidence and
  compare each bin's mean score to the humans' mean score in that bin. A judge
  that is right on average but wrong in every bin has good correlation and
  terrible calibration; ECE is what catches it.
* **Cohen's kappa** — agreement on the pass/fail decision, corrected for the
  agreement you would get by chance. Raw agreement looks excellent on any
  dataset where 90% of cases pass, which is most of them.
* **A linear recalibration** (slope and intercept) fitted by least squares, so
  future judge scores can be mapped toward the human scale.

Judges are then weighted by ``1 / (1 + ECE)`` when their scores are combined: a
well-calibrated judge counts for close to one vote, a poorly calibrated one for
a fraction of a vote, and no judge is ever silently discarded.

Example:
    >>> ece([0.9, 0.9, 0.1], [1.0, 1.0, 0.0], bins=2)
    0.1
    >>> cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0])
    1.0
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)

#: Number of reliability bins. Ten is the convention in the calibration
#: literature and keeps each bin populated at the sample sizes we label.
DEFAULT_BINS = 10

#: Below this many human labels the numbers are noise. The run still records
#: them, flagged, because "we have 12 labels" is more useful to a reviewer than
#: a missing row.
MIN_RELIABLE_SAMPLE = 30

#: A judge score at or above this is a "pass" for the kappa calculation.
PASS_THRESHOLD = 0.7


@dataclass(frozen=True, slots=True)
class Calibration:
    """One judge's calibration against human labels."""

    judge_model: str
    sample_size: int
    expected_calibration_error: float
    cohens_kappa: float | None
    pearson_r: float | None
    mean_absolute_error: float
    slope: float
    intercept: float
    weight: float
    reliability_bins: dict[str, Any]

    @property
    def is_reliable(self) -> bool:
        """Whether the sample is large enough to act on.

        Example:
            >>> Calibration("j", 5, 0.1, None, None, 0.1, 1.0, 0.0, 0.9, {}).is_reliable
            False
        """
        return self.sample_size >= MIN_RELIABLE_SAMPLE

    def apply(self, score: float) -> float:
        """Map a raw judge score onto the human scale.

        Clamped to [0, 1]: a fitted line can leave the interval at the extremes,
        and a "groundedness of 1.04" is not a number anyone should have to
        explain in a report.

        Example:
            >>> Calibration("j", 50, 0.1, None, None, 0.1, 0.5, 0.1, 0.9, {}).apply(1.0)
            0.6
        """
        return round(min(1.0, max(0.0, self.slope * score + self.intercept)), 6)


def ece(
    judge_scores: Sequence[float], human_scores: Sequence[float], *, bins: int = DEFAULT_BINS
) -> float:
    """Expected calibration error between judge and human scores.

    Each bin contributes its |mean judge - mean human| weighted by how many
    samples fall in it, so a bin holding two cases cannot dominate one holding
    forty.

    Raises:
        ValueError: when the two sequences differ in length. Silently zipping
            them would drop the tail of the longer one and quietly compute the
            wrong number.

    Example:
        >>> ece([0.5], [0.5])
        0.0
        >>> round(ece([1.0, 1.0], [0.0, 0.0]), 3)
        1.0
    """
    _require_paired(judge_scores, human_scores)
    if not judge_scores:
        return 0.0

    total = len(judge_scores)
    error = 0.0
    for lower, upper in _bin_edges(bins):
        members = [
            (j, h)
            for j, h in zip(judge_scores, human_scores, strict=True)
            if lower <= j < upper or (upper == 1.0 and j == 1.0)
        ]
        if not members:
            continue
        judge_mean = sum(j for j, _ in members) / len(members)
        human_mean = sum(h for _, h in members) / len(members)
        error += (len(members) / total) * abs(judge_mean - human_mean)

    return round(error, 6)


def reliability_bins(
    judge_scores: Sequence[float], human_scores: Sequence[float], *, bins: int = DEFAULT_BINS
) -> dict[str, Any]:
    """Per-bin means and counts, for the reliability diagram in the admin UI.

    The diagram is the part a reviewer actually reads: a single ECE number says
    the judge is off, the diagram says *where* — over-confident at the top,
    under-confident in the middle, or noisy throughout.

    Example:
        >>> reliability_bins([0.05], [0.0], bins=2)["bins"][0]["count"]
        1
    """
    _require_paired(judge_scores, human_scores)
    out: list[dict[str, Any]] = []
    for lower, upper in _bin_edges(bins):
        members = [
            (j, h)
            for j, h in zip(judge_scores, human_scores, strict=True)
            if lower <= j < upper or (upper == 1.0 and j == 1.0)
        ]
        out.append(
            {
                "lower": round(lower, 4),
                "upper": round(upper, 4),
                "count": len(members),
                "judge_mean": round(sum(j for j, _ in members) / len(members), 4)
                if members
                else None,
                "human_mean": round(sum(h for _, h in members) / len(members), 4)
                if members
                else None,
            }
        )
    return {"bins": out, "bin_count": bins}


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float | None:
    """Cohen's kappa for two binary raters.

    Returns ``None`` when chance agreement is total — both raters gave the same
    label to everything — because kappa is undefined there rather than perfect.
    Reporting 1.0 in that case would claim strong agreement from a dataset that
    demonstrates none.

    Example:
        >>> cohens_kappa([1, 0, 1, 0], [1, 0, 0, 1])
        0.0
        >>> cohens_kappa([1, 1], [1, 1]) is None
        True
    """
    _require_paired(a, b)
    n = len(a)
    if n == 0:
        return None

    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    p_a1 = sum(a) / n
    p_b1 = sum(b) / n
    expected = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)

    if math.isclose(expected, 1.0):
        return None
    return round((observed - expected) / (1 - expected), 6)


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson correlation, or ``None`` when either side has no variance.

    Example:
        >>> pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        1.0
        >>> pearson_r([1.0, 1.0], [1.0, 2.0]) is None
        True
    """
    _require_paired(x, y)
    n = len(x)
    if n < 2:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n
    dx = [v - mean_x for v in x]
    dy = [v - mean_y for v in y]

    denominator = math.sqrt(sum(v * v for v in dx)) * math.sqrt(sum(v * v for v in dy))
    if math.isclose(denominator, 0.0):
        return None
    return round(sum(a * b for a, b in zip(dx, dy, strict=True)) / denominator, 6)


def fit_line(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Least-squares slope and intercept mapping ``x`` onto ``y``.

    Falls back to the identity when the judge gave every case the same score:
    there is no line to fit through a vertical point cloud, and the identity at
    least leaves the scores unchanged instead of collapsing them to a constant.

    Example:
        >>> fit_line([0.0, 1.0], [0.0, 0.5])
        (0.5, 0.0)
        >>> fit_line([0.5, 0.5], [0.1, 0.9])
        (1.0, 0.0)
    """
    _require_paired(x, y)
    n = len(x)
    if n < 2:
        return (1.0, 0.0)

    mean_x = sum(x) / n
    mean_y = sum(y) / n
    variance = sum((v - mean_x) ** 2 for v in x)
    if math.isclose(variance, 0.0):
        return (1.0, 0.0)

    slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True)) / variance
    return (round(slope, 6), round(mean_y - slope * mean_x, 6))


def calibrate(
    judge_model: str, judge_scores: Sequence[float], human_scores: Sequence[float]
) -> Calibration:
    """Compute a judge's full calibration.

    Example:
        >>> c = calibrate("gpt", [0.8, 0.6, 0.9, 0.2], [0.8, 0.6, 0.9, 0.2])
        >>> c.expected_calibration_error, c.weight
        (0.0, 1.0)
    """
    _require_paired(judge_scores, human_scores)
    error = ece(judge_scores, human_scores)
    slope, intercept = fit_line(judge_scores, human_scores)

    return Calibration(
        judge_model=judge_model,
        sample_size=len(judge_scores),
        expected_calibration_error=error,
        cohens_kappa=cohens_kappa(
            [int(s >= PASS_THRESHOLD) for s in judge_scores],
            [int(s >= PASS_THRESHOLD) for s in human_scores],
        ),
        pearson_r=pearson_r(judge_scores, human_scores),
        mean_absolute_error=round(
            sum(abs(j - h) for j, h in zip(judge_scores, human_scores, strict=True))
            / len(judge_scores),
            6,
        )
        if judge_scores
        else 0.0,
        slope=slope,
        intercept=intercept,
        weight=weight_for(error),
        reliability_bins=reliability_bins(judge_scores, human_scores),
    )


def weight_for(calibration_error: float) -> float:
    """Convert an ECE into a voting weight.

    ``1 / (1 + ECE)`` is monotonic, bounded in (0, 1], and degrades gently: a
    judge with an ECE of 0.1 keeps 91% of its vote, one at 0.5 keeps 67%. A
    sharper penalty would make the combined score hostage to whichever judge
    happened to calibrate best on a small sample.

    Example:
        >>> weight_for(0.0), weight_for(1.0)
        (1.0, 0.5)
    """
    return round(1.0 / (1.0 + max(0.0, calibration_error)), 6)


def combine(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Combine judge scores by calibration weight.

    An unknown judge is weighted 1.0 rather than dropped — a judge added this
    week has no calibration yet, and excluding it would silently change what the
    metric means mid-experiment.

    Example:
        >>> combine({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 1.0})
        0.5
        >>> combine({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0})
        1.0
        >>> combine({}, {})
        0.0
    """
    if not scores:
        return 0.0

    total_weight = sum(weights.get(judge, 1.0) for judge in scores)
    if math.isclose(total_weight, 0.0):
        # Every judge is worthless; an unweighted mean is more honest than a
        # division by zero or a fabricated zero score.
        return round(sum(scores.values()) / len(scores), 6)

    return round(
        sum(score * weights.get(judge, 1.0) for judge, score in scores.items()) / total_weight,
        6,
    )


async def recalibrate(*, set_version: str = "v1") -> dict[str, Any]:
    """Recompute every judge's calibration from stored human labels.

    Run weekly by the beat schedule. Calibration drifts because the judge model
    changes underneath us — a provider's silent update can move a judge's scores
    without any change on our side — so a calibration measured once and trusted
    forever is worse than none, because it carries authority it no longer earns.
    """
    from sqlalchemy import select

    from src.core.db import system_session
    from src.models.evaluation import EvalCaseResult, JudgeCalibration

    async with system_session(reason="weekly judge recalibration") as session:
        labelled = (
            (
                await session.execute(
                    select(EvalCaseResult).where(EvalCaseResult.human_label.is_not(None))
                )
            )
            .scalars()
            .all()
        )

        by_judge: dict[str, tuple[list[float], list[float]]] = {}
        for case in labelled:
            for judge, raw in (case.judge_scores or {}).items():
                score = _score_of(raw)
                if score is None:
                    continue
                judge_scores, human_scores = by_judge.setdefault(judge, ([], []))
                judge_scores.append(score)
                human_scores.append(float(case.human_label or 0.0))

        results: dict[str, Any] = {}
        for judge, (judge_scores, human_scores) in by_judge.items():
            calibration = calibrate(judge, judge_scores, human_scores)
            if not calibration.is_reliable:
                log.warning(
                    "judge calibration is based on too few human labels to act on",
                    judge=judge,
                    sample_size=calibration.sample_size,
                    minimum=MIN_RELIABLE_SAMPLE,
                )

            # Previous calibrations are retired rather than updated, so the
            # history stays auditable: a reviewer can see when a judge's
            # calibration moved and correlate it with a metric shift.
            for previous in (
                (
                    await session.execute(
                        select(JudgeCalibration).where(
                            JudgeCalibration.judge_model == judge,
                            JudgeCalibration.is_active.is_(True),
                        )
                    )
                )
                .scalars()
                .all()
            ):
                previous.is_active = False

            session.add(
                JudgeCalibration(
                    judge_model=judge,
                    calibration_set_version=set_version,
                    sample_size=calibration.sample_size,
                    expected_calibration_error=calibration.expected_calibration_error,
                    cohens_kappa=calibration.cohens_kappa,
                    pearson_r=calibration.pearson_r,
                    mean_absolute_error=calibration.mean_absolute_error,
                    slope=calibration.slope,
                    intercept=calibration.intercept,
                    weight=calibration.weight,
                    reliability_bins=calibration.reliability_bins,
                    computed_at=datetime.now(UTC),
                    is_active=True,
                )
            )
            results[judge] = {
                "sample_size": calibration.sample_size,
                "ece": calibration.expected_calibration_error,
                "kappa": calibration.cohens_kappa,
                "weight": calibration.weight,
                "reliable": calibration.is_reliable,
            }

    log.info("recalibrated judges", judges=len(results))
    return {"judges": results, "labelled_cases": len(labelled)}


async def active_weights() -> dict[str, float]:
    """Load the current calibration weight for each judge."""
    from sqlalchemy import select

    from src.core.db import system_session
    from src.models.evaluation import JudgeCalibration

    async with system_session(reason="load judge weights") as session:
        rows = (
            (
                await session.execute(
                    select(JudgeCalibration).where(JudgeCalibration.is_active.is_(True))
                )
            )
            .scalars()
            .all()
        )
    return {row.judge_model: float(row.weight) for row in rows}


def _score_of(raw: Any) -> float | None:
    """Pull a numeric score out of a stored judge verdict.

    Verdicts are stored as whole objects (score plus reasoning), but older rows
    hold a bare number. Both shapes are read rather than migrated, because a
    migration that reshapes historical eval data destroys the comparability the
    data exists for.

    Example:
        >>> _score_of({"score": 0.8}), _score_of(0.5), _score_of("bad")
        (0.8, 0.5, None)
    """
    value = raw.get("score") if isinstance(raw, dict) else raw
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _bin_edges(bins: int) -> list[tuple[float, float]]:
    """Equal-width bin edges over [0, 1].

    Example:
        >>> _bin_edges(2)
        [(0.0, 0.5), (0.5, 1.0)]
    """
    width = 1.0 / bins
    return [(i * width, (i + 1) * width) for i in range(bins)]


def _require_paired(a: Sequence[Any], b: Sequence[Any]) -> None:
    """Reject unpaired sequences.

    Raises:
        ValueError: when the lengths differ.
    """
    if len(a) != len(b):
        msg = f"paired sequences must be the same length; got {len(a)} and {len(b)}"
        raise ValueError(msg)
