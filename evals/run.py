"""Run an eval set.

    python -m evals.run --set golden
    python -m evals.run --set adversarial --no-judge
    python -m evals.run --set regression --gate --baseline evals/baselines/main.json

This is the entry point the README points at and the CI gate calls. It has one
job beyond running the cases: making the numbers reproducible by anyone who
clones the repository. Every figure in the README and on the dashboards comes
from this command.

Two modes:

* **Live** (default) answers through the real API, which requires the stack to be
  up and provider keys to be set.
* **Offline** (``--offline``) answers with a deterministic stub. It measures
  nothing about answer quality and says so — it exists so that the harness, the
  gate and the report can be exercised in CI without a provider bill, and so a
  contributor can see the pipeline work before they have keys.

The distinction is loud on purpose. An offline run's report is stamped as such
and its numbers must never be quoted.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from evals import datasets
from evals.gate import evaluate, load_baseline, markdown_comment, save_baseline
from evals.report import write_all
from evals.runner import run_set
from evals.types import EvalCase

#: Resolved at import: Path.resolve() touches the filesystem, so it must not
#: run inside an async function.
_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASELINE = "evals/baselines/main.json"
DEFAULT_REPORTS = "evals/reports"

#: Two judges, from two providers. Same-family judges agree with the generator
#: for reasons that have nothing to do with the answer.
DEFAULT_JUDGES = ("claude-sonnet-5", "gpt-4o")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Example:
        >>> parse_args(["--set", "golden"]).set_name
        'golden'
    """
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__)
    parser.add_argument(
        "--set",
        dest="set_name",
        default="golden",
        choices=sorted(datasets.SETS),
        help="Which set to run.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases.")
    parser.add_argument("--concurrency", type=int, default=6, help="Cases in flight at once.")
    parser.add_argument("--model", default=None, help="Model under test.")
    parser.add_argument(
        "--judges",
        default=",".join(DEFAULT_JUDGES),
        help="Comma-separated judge models. Empty disables judging.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Deterministic metrics only: free, fast, and cannot flake.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Answer with a stub instead of the real system. Exercises the harness only.",
    )
    parser.add_argument("--api-url", default=os.getenv("AGRAG_API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("AGRAG_API_KEY", ""))
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Compare against the baseline and exit non-zero on failure.",
    )
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument(
        "--update-baseline", action="store_true", help="Write this run as the new baseline."
    )
    parser.add_argument("--reports", default=DEFAULT_REPORTS)
    parser.add_argument("--comment-out", default=None, help="Write the PR comment markdown here.")
    parser.add_argument(
        "--enforce-size", action="store_true", help="Require the set's committed minimum size."
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    """Run the requested set and report.

    Returns:
        0 when the run succeeded and the gate (if requested) passed, 1 otherwise.
    """
    args = parse_args(argv)

    cases = datasets.load(args.set_name, enforce_size=args.enforce_size)
    if args.limit:
        cases = cases[: args.limit]

    summary = datasets.describe(cases)
    print(f"\n{args.set_name} · {summary['count']} cases")
    print(f"  intents:     {summary['by_intent']}")
    print(f"  difficulty:  {summary['by_difficulty']}")
    if summary["by_attack"]:
        print(f"  attacks:     {summary['by_attack']}")
    if args.offline:
        print("\n  !! OFFLINE MODE — answers are stubbed. These numbers measure nothing.\n")

    answer_fn, model, closer = await _build_answer_fn(args)
    judge_router, judge_models, weights = await _build_judges(args)

    def progress(done: int, total: int, result: Any) -> None:
        """One line per case, so a long run shows movement."""
        mark = "." if result.passed else "F"
        sys.stdout.write(mark)
        sys.stdout.flush()
        if done == total:
            sys.stdout.write("\n")

    try:
        run = await run_set(
            cases,
            answer_fn=answer_fn,
            judge_router=judge_router,
            judge_models=judge_models,
            judge_weights=weights,
            set_name=args.set_name,
            model=model,
            concurrency=args.concurrency,
            progress=progress,
        )
    finally:
        if closer is not None:
            await closer()

    payload = run.to_dict()
    payload["offline"] = args.offline

    gate_report = None
    if args.gate:
        baseline = load_baseline(args.baseline)
        gate_report = evaluate(
            run.metrics,
            baseline=(baseline or {}).get("metrics") if baseline else None,
            case_results={r.case.id: r.passed for r in run.results},
            baseline_cases=(baseline or {}).get("cases") if baseline else None,
        )

    paths = write_all(
        payload, directory=args.reports, gate=gate_report.to_dict() if gate_report else None
    )
    _print_summary(payload, gate_report)
    print(f"\nreport: {paths['html']}\n        {paths['json']}")

    if args.comment_out and gate_report is not None:
        await asyncio.to_thread(
            Path(args.comment_out).write_text,
            markdown_comment(payload, gate_report),
            encoding="utf-8",
        )
        print(f"comment: {args.comment_out}")

    if args.update_baseline:
        if args.offline:
            # An offline baseline would be a fabricated one, and every future
            # gate would compare real numbers against invented ones.
            print("refusing to write a baseline from an offline run")
            return 1
        print(f"baseline: {save_baseline(args.baseline, payload)}")

    return 0 if (gate_report is None or gate_report.passed) else 1


async def _build_answer_fn(args: argparse.Namespace) -> tuple[Any, str, Any]:
    """Build the callable that answers one case.

    Returns the function, the model name to record, and an optional async
    cleanup callable.
    """
    if args.offline:
        return _offline_answer, "offline-stub", None

    import httpx

    client = httpx.AsyncClient(
        base_url=args.api_url,
        timeout=120.0,
        headers={"Authorization": f"Bearer {args.api_key}"} if args.api_key else {},
    )

    async def answer(case: EvalCase) -> Any:
        """Ask the real API, through the same endpoint a user's client would."""
        response = await client.post(
            "/api/chat",
            json={
                "message": case.query,
                "history": list(case.conversation),
                "include_thinking": False,
            },
        )
        response.raise_for_status()
        return _Answer.from_api(response.json())

    return answer, args.model or "api-default", client.aclose


async def _build_judges(args: argparse.Namespace) -> tuple[Any, tuple[str, ...], dict[str, float]]:
    """Build the judge router, or return no judges."""
    if args.no_judge or args.offline or not args.judges.strip():
        return None, (), {}

    sys.path.insert(0, str(_REPO_ROOT / "apps" / "api"))
    from src.core.config import get_settings
    from src.main import _build_providers
    from src.services.llm.router import LLMRouter, ModelPolicy

    settings = get_settings()
    providers = _build_providers(settings)
    if not providers:
        print("  no provider keys configured; running without judges")
        return None, (), {}

    models = tuple(m.strip() for m in args.judges.split(",") if m.strip())
    router = LLMRouter(
        providers=providers,
        policy=ModelPolicy(default_model=models[0], allowed_models=models),
    )

    weights: dict[str, float] = {}
    try:
        from src.services.calibration import active_weights

        weights = await active_weights()
    except Exception as exc:  # noqa: BLE001 - uncalibrated judges still vote, at 1.0
        print(f"  no judge calibration available ({exc}); judges weigh equally")

    return router, models, weights


class _Answer:
    """The shape the runner expects from an answer."""

    def __init__(self, **fields: Any) -> None:
        """Store the fields the runner reads."""
        self.content: str = fields.get("content", "")
        self.citation_sources: tuple[str, ...] = tuple(fields.get("citation_sources", ()))
        self.retrieved_sources: tuple[str, ...] = tuple(fields.get("retrieved_sources", ()))
        self.context: tuple[str, ...] = tuple(fields.get("context", ()))
        self.cost_usd: float = float(fields.get("cost_usd", 0.0))
        self.trace_id: str | None = fields.get("trace_id")
        self.stop_reason: str | None = fields.get("stop_reason")

    @classmethod
    def from_api(cls, body: dict[str, Any]) -> _Answer:
        """Map the chat API's response onto the runner's expectations."""
        citations = body.get("citations") or []
        return cls(
            content=body.get("content", ""),
            citation_sources=[
                c.get("document_title") or c.get("document_id", "") for c in citations
            ],
            retrieved_sources=body.get("retrieved_sources", []),
            context=[c.get("snippet", "") for c in citations],
            cost_usd=body.get("cost_usd", 0.0),
            trace_id=body.get("trace_id"),
            stop_reason=body.get("stop_reason"),
        )


async def _offline_answer(case: EvalCase) -> _Answer:
    """A stub answer, for exercising the harness without a provider.

    It refuses on cases that expect a refusal and echoes the expected answer
    otherwise, which makes the harness produce a full green run. That is
    precisely why an offline run's numbers are meaningless and are stamped as
    such in the report.
    """
    if case.expects_refusal or not case.expected_sources:
        return _Answer(content="I don't have information about that in the provided documents.")
    return _Answer(
        content=(case.expected_answer or "No answer available.") + " [1]",
        citation_sources=list(case.expected_sources),
        retrieved_sources=list(case.expected_sources),
        context=[case.expected_answer or ""],
    )


def _print_summary(run: dict[str, Any], gate: Any) -> None:
    """Print the headline numbers."""
    print(f"\n{run['passed']}/{run['case_count']} passed ({run['pass_rate']:.1%})")
    print(f"cost ${run['total_cost_usd']:.4f} · {run['duration_seconds']:.0f}s\n")

    for name, value in sorted(run.get("metrics", {}).items()):
        print(f"  {name:<28} {value:.4f}")

    modes: dict[str, int] = {}
    for result in run.get("results", []):
        if not result.get("passed"):
            mode = result.get("failure_mode") or "unknown"
            modes[mode] = modes.get(mode, 0) + 1
    if modes:
        print("\nfailure modes:")
        for mode, count in sorted(modes.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>4}  {mode}")

    if gate is not None:
        print(f"\ngate: {'PASS' if gate.passed else 'FAIL'}")
        for failure in gate.failures:
            print(f"  FAIL {failure}")
        for warning in gate.warnings:
            print(f"  warn {warning}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
