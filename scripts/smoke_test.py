"""Prove a deployment actually works, end to end.

    python scripts/smoke_test.py --url https://demo.example.com

The last check in the deploy pipeline. Everything before it can pass while the
demo returns a 500 to the first person who opens it: the image built, the
migration applied, the health check answered, and the retrieval path is broken
because an index was never created.

So this asks a real question and requires a real, cited answer.
"""

from __future__ import annotations

import argparse
import sys

import httpx

#: A question the demo corpus definitely answers, with the value the answer must
#: contain. Checking only for a 200 would pass on "I don't know".
PROBE = ("What is the carry-over limit for annual leave?", "10 days")

TIMEOUT = 60.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    failures: list[str] = []

    with httpx.Client(timeout=TIMEOUT, headers=headers, follow_redirects=True) as client:
        failures += check_health(client, base)
        failures += check_answer(client, base)

    if failures:
        print("\nSMOKE TEST FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nsmoke test passed")
    return 0


def check_health(client: httpx.Client, base: str) -> list[str]:
    """Liveness and readiness."""
    failures = []
    for path in ("/healthz", "/readyz"):
        try:
            response = client.get(f"{base}{path}")
            ok = response.status_code == 200
            print(f"  {path:<10} {response.status_code}")
            if not ok:
                failures.append(f"{path} returned {response.status_code}")
        except httpx.HTTPError as exc:
            failures.append(f"{path} could not be reached: {exc}")
    return failures


def check_answer(client: httpx.Client, base: str) -> list[str]:
    """Ask a real question and require a grounded, cited answer."""
    question, expected = PROBE
    try:
        response = client.post(f"{base}/api/chat", json={"message": question})
    except httpx.HTTPError as exc:
        return [f"the chat endpoint could not be reached: {exc}"]

    if response.status_code != 200:
        return [f"the chat endpoint returned {response.status_code}: {response.text[:200]}"]

    body = response.json()
    content = str(body.get("content", ""))
    citations = body.get("citations") or []
    print(f"  /api/chat  200  {len(content)} chars, {len(citations)} citations")

    failures = []
    if not content.strip():
        failures.append("the answer was empty")
    if expected.lower() not in content.lower():
        # Not a style preference: if the deployment cannot retrieve a fact the
        # corpus definitely contains, retrieval is broken however good the prose.
        failures.append(f"the answer did not contain {expected!r}: {content[:200]!r}")
    if not citations:
        failures.append("the answer cited nothing")
    return failures


if __name__ == "__main__":
    sys.exit(main())
