"""Watch a canary stage and fail if it is worse than the release it replaces.

    python scripts/watch_canary.py --stage 10 --minutes 5

Promotion is a decision, not a delay. Sleeping between rollout steps and hoping
is the common pattern and it catches nothing: by the time somebody looks at a
dashboard, 100% of traffic is already on the bad build.

Three conditions fail a stage. Each is a thing users would actually notice:

* the error rate exceeds ERROR_RATE_LIMIT,
* p95 time-to-first-token exceeds TTFT_LIMIT_MS,
* the canary serves no traffic at all — which usually means it is crash-looping
  and the load balancer has taken it out, so a green error rate is meaningless.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

ERROR_RATE_LIMIT = 0.01
TTFT_LIMIT_MS = 3000
POLL_SECONDS = 30

QUERIES = {
    "error_rate": 'sum(rate(http_requests_total{status=~"5..",release="canary"}[2m])) '
    '/ clamp_min(sum(rate(http_requests_total{release="canary"}[2m])), 0.001)',
    "ttft_p95": 'histogram_quantile(0.95, sum(rate(rag_ttft_ms_bucket{release="canary"}[2m])) by (le))',
    "requests": 'sum(rate(http_requests_total{release="canary"}[2m]))',
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True, help="Traffic percentage.")
    parser.add_argument("--minutes", type=int, default=5)
    parser.add_argument("--prometheus", default=None)
    args = parser.parse_args()

    import os

    prometheus = args.prometheus or os.getenv("PROMETHEUS_URL")
    if not prometheus:
        # Without metrics there is nothing to decide on. Passing anyway would
        # turn the gate into a sleep that always succeeds.
        print("PROMETHEUS_URL is not set; cannot evaluate the canary")
        return 1

    deadline = time.time() + args.minutes * 60
    print(f"watching the {args.stage}% canary for {args.minutes} minutes")

    with httpx.Client(timeout=15.0) as client:
        while time.time() < deadline:
            metrics = {name: query(client, prometheus, q) for name, q in QUERIES.items()}
            remaining = int(deadline - time.time())
            print(
                f"  error={metrics['error_rate']:.4f} "
                f"ttft_p95={metrics['ttft_p95']:.0f}ms "
                f"rps={metrics['requests']:.2f} "
                f"({remaining}s left)"
            )

            if metrics["error_rate"] > ERROR_RATE_LIMIT:
                print(f"FAIL: error rate {metrics['error_rate']:.4f} > {ERROR_RATE_LIMIT}")
                return 1
            if metrics["ttft_p95"] > TTFT_LIMIT_MS:
                print(f"FAIL: p95 TTFT {metrics['ttft_p95']:.0f}ms > {TTFT_LIMIT_MS}ms")
                return 1

            time.sleep(POLL_SECONDS)

    final = query(client_none := httpx.Client(timeout=15.0), prometheus, QUERIES["requests"])
    client_none.close()
    if final <= 0:
        print("FAIL: the canary served no traffic; it is probably not healthy")
        return 1

    print(f"the {args.stage}% stage looks healthy")
    return 0


def query(client: httpx.Client, prometheus: str, expression: str) -> float:
    """Run one instant query, returning 0.0 when there is no data.

    An absent series is not an error: a canary that has served no 5xx has no
    error-rate series at all, and treating that as a failure would block every
    healthy deploy.
    """
    try:
        response = client.get(
            f"{prometheus.rstrip('/')}/api/v1/query", params={"query": expression}
        )
        response.raise_for_status()
        result = response.json()["data"]["result"]
    except Exception as exc:  # noqa: BLE001 - a scrape failure is not a verdict
        print(f"    (could not query Prometheus: {exc})")
        return 0.0

    if not result:
        return 0.0
    return float(result[0]["value"][1])


if __name__ == "__main__":
    sys.exit(main())
