# syntax=docker/dockerfile:1.7
# code-exec MCP server. Published as an OCI image; see README.md.
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1     PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl     && rm -rf /var/lib/apt/lists/*     && useradd --create-home --uid 10001 mcp

WORKDIR /app

COPY common/ /app/common/
COPY code-exec/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY code-exec/ /app/

ENV PYTHONPATH=/app:/app/common

# Runs unprivileged with a read-only root filesystem (set in the compose file and
# the Helm chart). This server executes model-written code, so its container is the real
# security boundary: deploy it with no network egress and no mounted volumes.
USER mcp

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3   CMD curl -sf http://localhost:8080/healthz || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
