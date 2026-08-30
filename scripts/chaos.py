"""Fault injection.

    python scripts/chaos.py                 # every scenario
    python scripts/chaos.py --only redis
    python scripts/chaos.py --list

"Degrades gracefully" is a claim. This turns it into a fact, or into a bug.

Each scenario kills a dependency, asserts the system still does what it claims,
and restores it. The assertions are specific: after killing OpenSearch the API
must still answer *and* `/readyz` must report sparse retrieval as unavailable.
A system that keeps answering while claiming to be fully healthy is lying, and
that is worse than one that fails cleanly — the on-call engineer trusts the
health check.

Requires Docker and the compose stack. It stops containers, so do not point it
at anything you care about; it refuses to run against a non-local API for
exactly that reason.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass

import httpx

API = "http://localhost:8000"

#: A question the demo corpus answers, so a degraded-but-working system can be
#: distinguished from a broken one.
PROBE = "What is the carry-over limit for annual leave?"

#: How long to wait for the system to notice a dependency has gone. Health
#: checks and connection pools do not fail instantly, and asserting immediately
#: measures the timeout rather than the behaviour.
SETTLE_SECONDS = 8

RESTORE_SECONDS = 12


@dataclass(frozen=True)
class Scenario:
    """One fault and what must remain true while it is injected."""

    name: str
    container: str
    description: str
    expect_answers: bool
    degraded_capability: str | None = None


SCENARIOS = (
    Scenario(
        name="redis",
        container="agrag-redis",
        description="Caching, rate limits and live progress go; answers keep working",
        expect_answers=True,
        degraded_capability="cache",
    ),
    Scenario(
        name="opensearch",
        container="agrag-opensearch",
        description="Retrieval degrades to dense-only",
        expect_answers=True,
        degraded_capability="sparse_retrieval",
    ),
    Scenario(
        name="neo4j",
        container="agrag-neo4j",
        description="GraphRAG unavailable; other strategies unaffected",
        expect_answers=True,
        degraded_capability="graph",
    ),
    Scenario(
        name="minio",
        container="agrag-minio",
        description="Uploads fail; existing chunks still answer",
        expect_answers=True,
        degraded_capability="object_storage",
    ),
    Scenario(
        name="worker",
        container="agrag-worker",
        description="Ingestion queues rather than failing; answers unaffected",
        expect_answers=True,
    ),
    Scenario(
        name="postgres",
        container="agrag-postgres",
        # The one hard dependency. It is in the list because "fails cleanly" is
        # also a behaviour worth asserting: the API must return a 5xx with a
        # useful message, not hang until the client times out.
        description="The one hard dependency: must fail fast and clearly",
        expect_answers=False,
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="Run one scenario by name.")
    parser.add_argument("--list", action="store_true", help="List the scenarios and exit.")
    parser.add_argument("--api", default=API)
    args = parser.parse_args()

    if args.list:
        for scenario in SCENARIOS:
            print(f"  {scenario.name:<12} {scenario.description}")
        return 0

    if "localhost" not in args.api and "127.0.0.1" not in args.api:
        print("refusing to inject faults against a non-local API", file=sys.stderr)
        return 2

    scenarios = [s for s in SCENARIOS if args.only in (None, s.name)]
    if not scenarios:
        print(f"no scenario named {args.only!r}", file=sys.stderr)
        return 2

    print("baseline")
    if not _healthy(args.api):
        print("  the stack is not healthy; start it with `docker compose up` first")
        return 1
    print("  ok\n")

    failures: list[str] = []
    for scenario in scenarios:
        failures += run_scenario(scenario, api=args.api)

    print()
    if failures:
        print(f"{len(failures)} assertion(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("every scenario behaved as documented")
    return 0


def run_scenario(scenario: Scenario, *, api: str) -> list[str]:
    """Inject one fault, assert, and restore.

    Restoration happens in a `finally` so an assertion failure does not leave the
    stack broken. A chaos script that only cleans up on success is one that
    leaves a developer's machine in pieces the first time it finds something.
    """
    print(f"{scenario.name}: {scenario.description}")
    failures: list[str] = []

    try:
        _docker("stop", scenario.container)
        time.sleep(SETTLE_SECONDS)

        answered, detail = _can_answer(api)
        if scenario.expect_answers and not answered:
            failures.append(f"{scenario.name}: answers stopped ({detail})")
        elif not scenario.expect_answers and answered:
            failures.append(f"{scenario.name}: expected failure, but it answered")
        else:
            print(f"  answers: {'yes' if answered else 'no (expected)'}")

        if not scenario.expect_answers and not answered and "timed out" in detail:
            # Failing is correct; hanging is not. A client waiting 120 seconds
            # for a connection refusal is a worse outcome than a fast 503.
            failures.append(f"{scenario.name}: failed by hanging rather than erroring")

        if scenario.degraded_capability:
            reported = _readyz_reports_degraded(api, scenario.degraded_capability)
            if reported is None:
                failures.append(f"{scenario.name}: /readyz did not respond")
            elif not reported:
                failures.append(
                    f"{scenario.name}: /readyz still claims {scenario.degraded_capability} "
                    "is healthy — a health check that lies is worse than none"
                )
            else:
                print(f"  /readyz reports {scenario.degraded_capability} degraded")

    finally:
        _docker("start", scenario.container)
        time.sleep(RESTORE_SECONDS)

    if not _healthy(api):
        failures.append(f"{scenario.name}: the stack did not recover after restoration")
    else:
        print("  recovered\n")

    return failures


def _docker(action: str, container: str) -> None:
    """Stop or start a container."""
    subprocess.run(  # noqa: S603 - fixed argv, shell=False
        ["docker", action, container],  # noqa: S607 - resolved from PATH by design
        capture_output=True,
        check=False,
        timeout=60,
    )


def _healthy(api: str) -> bool:
    """Whether the API is answering its liveness probe."""
    try:
        return httpx.get(f"{api}/healthz", timeout=10).status_code == 200
    except httpx.HTTPError:
        return False


def _can_answer(api: str) -> tuple[bool, str]:
    """Whether the system still answers the probe question."""
    try:
        response = httpx.post(f"{api}/api/chat", json={"message": PROBE}, timeout=60)
    except httpx.TimeoutException:
        return False, "timed out"
    except httpx.HTTPError as exc:
        return False, str(exc)[:120]

    if response.status_code != 200:
        return False, f"status {response.status_code}"

    content = str(response.json().get("content", ""))
    return bool(content.strip()), "empty answer" if not content.strip() else "ok"


def _readyz_reports_degraded(api: str, capability: str) -> bool | None:
    """Whether `/readyz` admits the named capability is unavailable.

    Returns None when readiness could not be read at all, which the caller
    reports differently from "it lied".
    """
    try:
        response = httpx.get(f"{api}/readyz", timeout=10)
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return None

    checks = body.get("checks") or body.get("capabilities") or {}
    if capability not in checks:
        return None
    value = checks[capability]
    return value in {False, "unavailable", "degraded", "down"}


if __name__ == "__main__":
    sys.exit(main())
