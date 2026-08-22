"""Recompute judge calibration against the human labels.

    python -m evals.scripts.calibrate_judges              # report only
    python -m evals.scripts.calibrate_judges --write      # persist the result
    python -m evals.scripts.calibrate_judges --report calibration.md

Run weekly by `.github/workflows/judge-calibration.yml`.

The judges are language models, and a provider's silent update can move their
scores without any change on our side. A calibration measured once and trusted
forever is worse than no calibration at all, because it carries an authority it
no longer earns.

Exits 0 whether or not a judge drifted — drift is a finding, not a failure — but
sets the `drifted` GitHub output so the workflow can open an issue. Somebody has
to decide whether recent eval numbers, weighted by a calibration that no longer
holds, need re-running against a fresh baseline. That is a judgement call, not
something a cron job should make silently.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "api"))

#: An ECE moving by more than this since the last calibration counts as drift.
#: Below it, the movement is within what re-labelling a few dozen cases produces
#: on its own.
DRIFT_THRESHOLD = 0.05


async def run(*, write: bool, report_path: str | None) -> int:
    """Recalibrate, print the result, and flag drift."""
    from sqlalchemy import select
    from src.core.db import system_session
    from src.models.evaluation import JudgeCalibration
    from src.services.calibration import MIN_RELIABLE_SAMPLE, recalibrate

    async with system_session(reason="judge calibration report") as session:
        previous = {
            row.judge_model: row
            for row in (
                await session.execute(
                    select(JudgeCalibration).where(JudgeCalibration.is_active.is_(True))
                )
            )
            .scalars()
            .all()
        }

    if not write:
        # A dry run must not retire the active calibration rows, so it reads the
        # labels and computes without touching anything.
        result = await _dry_run()
    else:
        result = await recalibrate()

    judges = result.get("judges", {})
    labelled = result.get("labelled_cases", 0)

    print(f"\n{labelled} human-labelled cases across {len(judges)} judges\n")
    if not judges:
        print("No judge scores with human labels yet.")
        print("Label the disagreement queue at /admin -> Evals -> disagreements.")
        _emit_output(drifted=False)
        return 0

    drifted: list[str] = []
    lines = [
        "| Judge | n | ECE | Δ ECE | kappa | weight | reliable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]

    for judge, values in sorted(judges.items()):
        ece = float(values["ece"])
        before = previous.get(judge)
        delta = ece - float(before.expected_calibration_error) if before else None
        kappa = values.get("kappa")
        reliable = bool(values["reliable"])

        if delta is not None and abs(delta) > DRIFT_THRESHOLD:
            drifted.append(judge)

        print(
            f"  {judge:<28} n={values['sample_size']:<4} "
            f"ECE={ece:.4f}"
            + (f" ({delta:+.4f})" if delta is not None else " (first run)")
            + f"  kappa={kappa if kappa is None else f'{kappa:.3f}'}"
            + f"  weight={values['weight']:.3f}"
            + ("" if reliable else f"  [under {MIN_RELIABLE_SAMPLE} labels — not reliable]")
        )

        lines.append(
            f"| `{judge}` | {values['sample_size']} | {ece:.4f} | "
            + (f"{delta:+.4f}" if delta is not None else "—")
            + f" | {'—' if kappa is None else f'{kappa:.3f}'} "
            + f"| {values['weight']:.3f} | {'yes' if reliable else 'no'} |"
        )

    if drifted:
        print(f"\nDRIFT: {', '.join(drifted)} moved by more than {DRIFT_THRESHOLD}")
        print("Eval numbers since the last calibration were weighted by the old values.")
    else:
        print("\nNo judge drifted materially.")

    if not write:
        print("\n(dry run — nothing was written; pass --write to persist)")

    if report_path:
        Path(report_path).write_text(
            _markdown(lines, labelled=labelled, drifted=drifted, written=write),
            encoding="utf-8",
        )
        print(f"report: {report_path}")

    _emit_output(drifted=bool(drifted))
    return 0


async def _dry_run() -> dict[str, Any]:
    """Compute calibrations without writing them.

    Duplicates the aggregation in `recalibrate` rather than adding a flag to it,
    because a persistence function that can be asked not to persist is one call
    site away from a silent no-op in production.
    """
    from sqlalchemy import select
    from src.core.db import system_session
    from src.models.evaluation import EvalCaseResult
    from src.services.calibration import _score_of, calibrate

    async with system_session(reason="judge calibration dry run") as session:
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
            if judge.startswith("_"):
                continue
            score = _score_of(raw)
            if score is None:
                continue
            judge_scores, human_scores = by_judge.setdefault(judge, ([], []))
            judge_scores.append(score)
            human_scores.append(float(case.human_label or 0.0))

    judges = {}
    for judge, (judge_scores, human_scores) in by_judge.items():
        calibration = calibrate(judge, judge_scores, human_scores)
        judges[judge] = {
            "sample_size": calibration.sample_size,
            "ece": calibration.expected_calibration_error,
            "kappa": calibration.cohens_kappa,
            "weight": calibration.weight,
            "reliable": calibration.is_reliable,
        }

    return {"judges": judges, "labelled_cases": len(labelled)}


def _markdown(rows: list[str], *, labelled: int, drifted: list[str], written: bool) -> str:
    """Render the report for the GitHub issue."""
    body = [
        "## Judge calibration",
        "",
        f"Computed from **{labelled}** human-labelled eval cases.",
        "",
        *rows,
        "",
    ]

    if drifted:
        body += [
            f"### Drift: {', '.join(f'`{j}`' for j in drifted)}",
            "",
            f"These judges moved by more than {DRIFT_THRESHOLD} since the last "
            "calibration. Eval results recorded in between were weighted by the "
            "old values.",
            "",
            "**Decide:** re-run the baseline against the new weights, or accept "
            "the shift and note it. Either is defensible; leaving it undecided "
            "is not, because every subsequent gate compares against a baseline "
            "whose weighting no longer matches.",
            "",
        ]

    body += [
        "---",
        "",
        "`weight = 1 / (1 + ECE)` — see [docs/EVALS.md](../docs/EVALS.md) and "
        "[ADR 0003](../docs/adr/0003-two-judge-calibrated-panel.md).",
        "",
        "_Written by `python -m evals.scripts.calibrate_judges`"
        + ("" if written else " (dry run)")
        + "._",
    ]
    return "\n".join(body) + "\n"


def _emit_output(*, drifted: bool) -> None:
    """Set the `drifted` step output for the workflow."""
    path = os.getenv("GITHUB_OUTPUT")
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(f"drifted={'true' if drifted else 'false'}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist the calibration. Without it, nothing is written.",
    )
    parser.add_argument("--report", default=None, help="Write a markdown report here.")
    args = parser.parse_args()
    return asyncio.run(run(write=args.write, report_path=args.report))


if __name__ == "__main__":
    raise SystemExit(main())
