"""The batch API.

For work that is large and not interactive: re-answering a thousand questions
after a prompt change, backfilling answers for an eval set, bulk-processing a
customer's support archive. Submitting those one at a time through
``/v1/chat/completions`` would either rate-limit the caller or starve interactive
traffic behind them.

Batches run on their own Celery queue at a lower priority than live chat, so a
ten-thousand-item job cannot make the chat UI slow. Results are polled rather
than pushed, with an optional webhook when the job finishes.

A batch is **not** transactional. Individual items fail without failing the job,
and the result file records each item's outcome — the alternative, failing ten
thousand items because one question tripped a guardrail, is not useful to
anyone.

Example:
    >>> from src.api.routers.batch import router
    >>> router.prefix
    '/v1/batches'
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from src.api.auth import Principal, get_principal
from src.api.dependencies import AppServices, get_services
from src.core.errors import ConflictError, NotFoundError, ValidationFailedError
from src.core.logging import get_logger
from src.models.tenant import ApiKeyScope
from src.schemas.admin import BatchOut, BatchRequest

log = get_logger(__name__)

router = APIRouter(prefix="/v1/batches", tags=["batch"])

#: Batch state lives in Redis rather than Postgres. A batch is ephemeral
#: operational state with a known expiry, not a business record, and putting ten
#: thousand per-item results in a table would grow it without bound for data
#: nobody reads after a week.
BATCH_TTL_SECONDS = 7 * 24 * 3600

MAX_ITEMS = 10_000


@router.post("", response_model=BatchOut, status_code=202)
async def create_batch(
    request: BatchRequest,
    principal: Annotated[Principal, Depends(get_principal)],
    services: Annotated[AppServices, Depends(get_services)],
) -> BatchOut:
    """Submit a batch job.

    Returns 202 with a job id; poll ``GET /v1/batches/{id}`` for progress.

    Raises:
        ValidationFailedError: when custom ids repeat. They are how a caller
            matches results back to inputs, so duplicates would silently make
            some results unattributable.
        ConflictError: when Redis is unavailable — the batch would be accepted
            and then lost, which is worse than a clear refusal.
    """
    principal.require(ApiKeyScope.WRITE)

    custom_ids = [item.custom_id for item in request.requests]
    if len(set(custom_ids)) != len(custom_ids):
        msg = "custom_id must be unique within a batch"
        raise ValidationFailedError(msg)

    if services.redis is None:
        raise ConflictError("The batch API requires Redis, which is not available.")

    from src.models.base import new_id

    batch_id = new_id("batch")
    record = {
        "id": batch_id,
        "tenant_id": principal.tenant_id,
        "status": "queued",
        "total": len(request.requests),
        "completed": 0,
        "failed": 0,
        "created_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "metadata": dict(request.metadata),
        "items": [item.model_dump(mode="json") for item in request.requests],
        "results": [],
    }

    await services.redis.set(
        _key(principal.tenant_id, batch_id), json.dumps(record), ex=BATCH_TTL_SECONDS
    )

    from src.workers.celery_app import run_batch

    run_batch.delay(batch_id=batch_id, tenant_id=principal.tenant_id)
    log.info("queued a batch", batch_id=batch_id, items=len(request.requests))
    return _out(record)


@router.get("/{batch_id}", response_model=BatchOut)
async def get_batch(
    batch_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    services: Annotated[AppServices, Depends(get_services)],
) -> BatchOut:
    """Poll a batch's progress."""
    return _out(await _load(services, principal.tenant_id, batch_id))


@router.get("/{batch_id}/results")
async def get_batch_results(
    batch_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    services: Annotated[AppServices, Depends(get_services)],
    limit: Annotated[int, Query(ge=1, le=MAX_ITEMS)] = 1000,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """Read a batch's results, paged.

    Available while the batch is still running: for a long job the partial
    results are usually what the caller wants, and making them wait for the last
    item to fetch the first is needless.
    """
    record = await _load(services, principal.tenant_id, batch_id)
    results = record.get("results", [])
    return {
        "batch_id": batch_id,
        "status": record["status"],
        "total": record["total"],
        "returned": len(results[offset : offset + limit]),
        "results": results[offset : offset + limit],
    }


@router.post("/{batch_id}/cancel", response_model=BatchOut)
async def cancel_batch(
    batch_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    services: Annotated[AppServices, Depends(get_services)],
) -> BatchOut:
    """Cancel a batch.

    Items already finished keep their results; the worker checks the status
    between items and stops. Cancelling mid-item rather than between them would
    mean throwing away an answer that was already paid for.

    Raises:
        ConflictError: when the batch has already finished.
    """
    principal.require(ApiKeyScope.WRITE)

    record = await _load(services, principal.tenant_id, batch_id)
    if record["status"] in {"completed", "failed", "cancelled"}:
        raise ConflictError(f"That batch is already {record['status']}.")

    record["status"] = "cancelled"
    record["completed_at"] = datetime.now(UTC).isoformat()
    await services.redis.set(
        _key(principal.tenant_id, batch_id), json.dumps(record), ex=BATCH_TTL_SECONDS
    )
    log.info("cancelled a batch", batch_id=batch_id, completed=record["completed"])
    return _out(record)


async def _load(services: AppServices, tenant_id: str, batch_id: str) -> dict[str, Any]:
    """Load a batch record for this tenant.

    Scoped by tenant in the key itself, so one tenant cannot poll another's
    batch by guessing an id.

    Raises:
        NotFoundError: when there is no such batch, or it has expired.
        ConflictError: when Redis is unavailable.
    """
    if services.redis is None:
        raise ConflictError("The batch API requires Redis, which is not available.")

    raw = await services.redis.get(_key(tenant_id, batch_id))
    if raw is None:
        raise NotFoundError("Batch not found. Batches expire after seven days.")
    return json.loads(raw)


def _key(tenant_id: str, batch_id: str) -> str:
    """Redis key for a batch.

    Example:
        >>> _key("ten_1", "batch_2")
        'batch:ten_1:batch_2'
    """
    return f"batch:{tenant_id}:{batch_id}"


def _out(record: dict[str, Any]) -> BatchOut:
    """Map a stored batch record onto its API shape, without the item bodies."""
    return BatchOut(
        id=record["id"],
        status=record["status"],
        total=record["total"],
        completed=record["completed"],
        failed=record["failed"],
        created_at=datetime.fromisoformat(record["created_at"]),
        completed_at=datetime.fromisoformat(record["completed_at"])
        if record.get("completed_at")
        else None,
    )
