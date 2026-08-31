#!/usr/bin/env bash
# Block until the API healthcheck passes, so `make up` only returns on a live stack.
set -euo pipefail

URL="${1:-http://localhost:8000/healthz}"
TIMEOUT="${TIMEOUT:-300}"
start=$(date +%s)

printf 'waiting for %s ' "$URL"
until curl -sf "$URL" >/dev/null 2>&1; do
  now=$(date +%s)
  if (( now - start > TIMEOUT )); then
    printf '\ntimed out after %ss. Recent api logs:\n' "$TIMEOUT" >&2
    docker compose logs --tail 50 api >&2 || true
    exit 1
  fi
  printf '.'
  sleep 2
done
printf '\nstack healthy in %ss\n' "$(( $(date +%s) - start ))"
