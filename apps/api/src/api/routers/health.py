"""Health, readiness and metrics endpoints.

Liveness and readiness answer different questions and must not be conflated. A
liveness probe that checks the database restarts the API every time Postgres
blips — which is exactly when restarting helps least. So:

* ``/healthz`` — is this process alive? Answers from memory, always fast, never
  touches a dependency. Kubernetes restarts the pod when this fails.
* ``/readyz`` — can this process serve traffic? Checks the dependencies it
  genuinely cannot work without. Kubernetes removes the pod from the load
  balancer when this fails, and puts it back when it recovers.

Optional dependencies are reported as degraded rather than failing readiness: a
Neo4j outage should disable graph retrieval, not take chat offline.

Example:
    >>> from src.api.routers.health import router
    >>> router.tags
    ['health']
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from src.core.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["health"])

_STARTED_AT = time.time()

#: Dependencies without which the API cannot serve a request at all.
REQUIRED = ("database", "redis")

CHECK_TIMEOUT = 3.0


class HealthOut(BaseModel):
    """Liveness response."""

    model_config = ConfigDict(frozen=True)

    status: str = "ok"
    version: str
    uptime_seconds: float


class ReadinessOut(BaseModel):
    """Readiness response, with per-dependency detail."""

    model_config = ConfigDict(frozen=True)

    status: str
    checks: dict[str, str] = Field(default_factory=dict)
    degraded: list[str] = Field(default_factory=list)
    version: str


@router.get("/healthz", response_model=HealthOut)
async def healthz() -> HealthOut:
    """Liveness: is the process running?

    Deliberately checks nothing external. A liveness probe that depends on the
    database turns a database blip into a rolling restart of every pod.
    """
    return HealthOut(version="0.1.0", uptime_seconds=round(time.time() - _STARTED_AT, 1))


@router.get("/readyz", response_model=ReadinessOut)
async def readyz(request: Request, response: Response) -> ReadinessOut:
    """Readiness: can this process serve traffic?

    Returns 503 when a required dependency is unreachable, so the load balancer
    stops sending traffic here without the pod being restarted.
    """
    services = getattr(request.app.state, "services", None)
    if services is None:
        response.status_code = 503
        return ReadinessOut(status="starting", version="0.1.0")

    names, checks = _checks(services)
    results = await asyncio.gather(
        *(_run(name, check) for name, check in zip(names, checks, strict=True))
    )
    outcomes = dict(results)

    failed_required = [n for n in REQUIRED if outcomes.get(n, "unavailable") != "ok"]
    degraded = [n for n, status in outcomes.items() if status != "ok" and n not in REQUIRED]

    if failed_required:
        response.status_code = 503
        log.error("readiness failed", failed=failed_required)

    return ReadinessOut(
        status="ok" if not failed_required else "unavailable",
        checks=outcomes,
        degraded=sorted(degraded),
        version="0.1.0",
    )


def _checks(services: Any) -> tuple[list[str], list[Any]]:
    """Build the list of dependency checks to run."""
    names: list[str] = []
    checks: list[Any] = []

    names.append("database")
    checks.append(_check_database)

    names.append("redis")
    checks.append(lambda: _check_redis(services.redis))

    if services.opensearch is not None:
        names.append("opensearch")
        checks.append(lambda: _check_opensearch(services.opensearch))

    if services.neo4j is not None:
        names.append("neo4j")
        checks.append(lambda: _check_neo4j(services.neo4j))

    if services.storage is not None:
        names.append("storage")
        checks.append(lambda: _check_storage(services.storage))

    if services.tools is not None:
        names.append("mcp")
        checks.append(lambda: _check_tools(services.tools))

    return names, checks


async def _run(name: str, check: Any) -> tuple[str, str]:
    """Run one check with a timeout, never raising.

    A readiness endpoint that can hang is worse than one that reports a
    dependency down: the orchestrator gets no answer at all and eventually kills
    a pod that was merely waiting.
    """
    try:
        await asyncio.wait_for(check(), timeout=CHECK_TIMEOUT)
    except TimeoutError:
        return name, "timeout"
    except Exception as exc:  # noqa: BLE001 - the point is to report, not raise
        log.debug("readiness check failed", dependency=name, reason=str(exc))
        return name, "unavailable"
    return name, "ok"


async def _check_database() -> None:
    """Verify the database answers a trivial query."""
    from sqlalchemy import text

    from src.core.db import get_engine

    async with get_engine().connect() as connection:
        await connection.execute(text("SELECT 1"))


async def _check_redis(client: Any) -> None:
    """Verify Redis responds to PING."""
    if client is None:
        msg = "redis is not configured"
        raise RuntimeError(msg)
    await client.ping()


async def _check_opensearch(client: Any) -> None:
    """Verify the OpenSearch cluster is reachable."""
    await client.info()


async def _check_neo4j(driver: Any) -> None:
    """Verify Neo4j connectivity."""
    await driver.verify_connectivity()


async def _check_storage(client: Any) -> None:
    """Verify object storage is reachable."""
    await asyncio.to_thread(client.bucket_exists, client.default_bucket)


async def _check_tools(registry: Any) -> None:
    """Verify at least one MCP server is up.

    All servers being down disables tools but does not stop chat, so this only
    fails readiness if the registry itself is broken.
    """
    await registry.health()


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus scrape endpoint."""
    from src.observability.metrics import content_type, render

    return Response(content=render(), media_type=content_type())
