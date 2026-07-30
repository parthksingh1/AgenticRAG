"""Celery application and task definitions.

Ingestion is asynchronous because it is slow and bursty: a user dropping thirty
PDFs on the upload zone would otherwise hold thirty HTTP connections open for
minutes. The API enqueues and returns immediately; the worker does the work and
publishes progress.

Retries use exponential backoff with jitter, and a task that exhausts them is
**dead-lettered rather than lost**: the job row records the final error and the
document is marked failed with a message the user can act on. A silently
disappearing upload is the worst outcome here — the user sees a spinner that
never resolves and has no idea whether to retry.

Tasks are synchronous functions that drive the async pipeline through
``asyncio.run``. Celery's own async support is still awkward, and one event loop
per task is simpler to reason about than a shared loop across a prefork pool.

Example:
    >>> from src.workers.celery_app import celery_app
    >>> celery_app.main
    'agrag'
"""

from __future__ import annotations

import asyncio
from typing import Any

from celery import Celery
from celery.schedules import crontab

from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger

log = get_logger(__name__)
settings = get_settings()

celery_app = Celery(
    "agrag",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # One task at a time per worker process: ingestion is memory-hungry (a model
    # plus a document in flight), and prefetching several would let one large
    # document push a worker into swap.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    # Without this, a worker killed mid-task loses the message entirely, and the
    # user's upload disappears with no record.
    task_reject_on_worker_lost=True,
    task_track_started=True,
    task_time_limit=1800,
    task_soft_time_limit=1500,
    result_expires=86_400,
    task_routes={
        "agrag.ingest_document": {"queue": "ingestion"},
        "agrag.reindex_tenant": {"queue": "ingestion"},
        # Batches get their own queue so a ten-thousand-item job cannot sit in
        # front of an interactive ingestion.
        "agrag.run_batch": {"queue": "batch"},
    },
    beat_schedule={
        "nightly-backup": {
            "task": "agrag.backup",
            "schedule": crontab(hour=3, minute=0),
        },
        "daily-drift-snapshot": {
            "task": "agrag.snapshot_drift",
            "schedule": crontab(hour=2, minute=30),
        },
        "weekly-judge-calibration": {
            "task": "agrag.recalibrate_judges",
            "schedule": crontab(day_of_week=1, hour=4, minute=0),
        },
        "hourly-webhook-retry": {
            "task": "agrag.retry_webhooks",
            "schedule": crontab(minute=15),
        },
        "daily-retention-purge": {
            "task": "agrag.purge_expired",
            "schedule": crontab(hour=5, minute=0),
        },
    },
)


@celery_app.on_after_configure.connect
def _configure(sender: Any, **_kwargs: Any) -> None:
    """Give the worker the same structured logging as the API."""
    configure_logging(level=settings.log_level, json_output=settings.is_production)


@celery_app.task(
    name="agrag.ingest_document",
    bind=True,
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def ingest_document(self: Any, *, document_id: str, tenant_id: str) -> dict[str, Any]:
    """Ingest one document.

    Retries transient failures with backoff. A permanent failure — an unreadable
    file, an unsupported format — is recorded on the document and *not* retried,
    because retrying will produce the same error three more times and delay the
    user learning about it.
    """
    from src.core.errors import IngestionError, UnsupportedMediaTypeError

    try:
        return asyncio.run(_ingest(document_id=document_id, tenant_id=tenant_id))
    except (UnsupportedMediaTypeError, IngestionError) as exc:
        # The document itself is the problem; another attempt cannot help.
        asyncio.run(_record_failure(document_id, tenant_id, str(exc), permanent=True))
        return {"document_id": document_id, "status": "failed", "error": str(exc)}
    except Exception as exc:
        attempt = self.request.retries + 1
        exhausted = attempt > self.max_retries
        asyncio.run(
            _record_failure(document_id, tenant_id, str(exc), permanent=exhausted, attempt=attempt)
        )
        if exhausted:
            log.error(
                "ingestion dead-lettered after exhausting retries",
                document_id=document_id,
                attempts=attempt,
            )
            return {"document_id": document_id, "status": "dead_lettered", "error": str(exc)}
        raise


async def _ingest(*, document_id: str, tenant_id: str) -> dict[str, Any]:
    """Run the pipeline inside a tenant-scoped session."""
    from src.core.context import request_context
    from src.core.db import session_scope
    from src.ingestion.embedders.base import CachingEmbedder, InMemoryEmbeddingCache
    from src.services.ingestion import IngestionPipeline, ProgressReporter
    from src.services.storage import ObjectStorage

    with request_context(tenant_id=tenant_id):
        embedder = _worker_embedder()
        pipeline = IngestionPipeline(
            embedder=CachingEmbedder(embedder, cache=InMemoryEmbeddingCache()),
            storage=ObjectStorage(settings),
            sparse=await _worker_sparse(),
            progress=ProgressReporter(await _worker_redis()),
            max_bytes=settings.max_upload_bytes,
        )
        async with session_scope() as session:
            result = await pipeline.run(session, document_id=document_id, tenant_id=tenant_id)

    return {
        "document_id": result.document_id,
        "status": "indexed",
        "chunks": result.chunks_created,
        "deduplicated": result.chunks_deduplicated,
        "seconds": result.duration_seconds,
    }


async def _record_failure(
    document_id: str, tenant_id: str, error: str, *, permanent: bool, attempt: int = 1
) -> None:
    """Record a failure on the document and its job.

    The message reaches the user, so it is trimmed rather than dumped: a
    thousand-line traceback in the document list is not actionable.
    """
    from sqlalchemy import select

    from src.core.context import request_context
    from src.core.db import session_scope
    from src.models.document import Document, DocumentStatus, IngestionJob
    from src.observability.metrics import record_ingestion

    message = error.strip().splitlines()[0][:500] if error.strip() else "ingestion failed"

    with request_context(tenant_id=tenant_id):
        async with session_scope() as session:
            document = (
                await session.execute(select(Document).where(Document.id == document_id))
            ).scalar_one_or_none()
            if document is not None and permanent:
                document.status = DocumentStatus.FAILED
                document.error_message = message

            job = (
                await session.execute(
                    select(IngestionJob)
                    .where(IngestionJob.document_id == document_id)
                    .order_by(IngestionJob.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if job is not None:
                job.attempts = attempt
                job.last_error = message
                if permanent:
                    job.status = DocumentStatus.FAILED
                    job.dead_lettered_at = _now()

    if permanent:
        record_ingestion(status="failed")
    log.error("ingestion failed", document_id=document_id, permanent=permanent, error=message)


@celery_app.task(name="agrag.reindex_tenant")
def reindex_tenant(*, tenant_id: str) -> dict[str, Any]:
    """Re-ingest every document for a tenant.

    Needed after an embedding-model change: existing vectors are in a different
    space and comparing across them produces confident nonsense.
    """
    return asyncio.run(_reindex(tenant_id))


async def _reindex(tenant_id: str) -> dict[str, Any]:
    """Queue one ingestion task per document."""
    from sqlalchemy import select

    from src.core.context import request_context
    from src.core.db import session_scope
    from src.models.document import Document

    with request_context(tenant_id=tenant_id):
        async with session_scope() as session:
            result = await session.execute(select(Document.id).where(Document.deleted_at.is_(None)))
            ids = [row[0] for row in result.all()]

    for document_id in ids:
        ingest_document.delay(document_id=document_id, tenant_id=tenant_id)

    log.warning("queued a full tenant reindex", tenant_id=tenant_id, documents=len(ids))
    return {"tenant_id": tenant_id, "queued": len(ids)}


@celery_app.task(name="agrag.run_batch")
def run_batch(*, batch_id: str, tenant_id: str) -> dict[str, Any]:
    """Answer every item in a batch.

    Items fail independently: one question tripping a guardrail must not fail the
    other nine thousand nine hundred and ninety nine.
    """
    return asyncio.run(_run_batch(batch_id=batch_id, tenant_id=tenant_id))


async def _run_batch(*, batch_id: str, tenant_id: str) -> dict[str, Any]:
    """Drive one batch, checkpointing progress after every item."""
    import json

    from src.core.context import request_context

    redis = await _worker_redis()
    if redis is None:
        log.error("cannot run a batch without Redis", batch_id=batch_id)
        return {"batch_id": batch_id, "status": "failed", "error": "redis unavailable"}

    key = f"batch:{tenant_id}:{batch_id}"
    raw = await redis.get(key)
    if raw is None:
        log.warning("batch record expired before it ran", batch_id=batch_id)
        return {"batch_id": batch_id, "status": "failed", "error": "expired"}

    record = json.loads(raw)
    record["status"] = "running"
    await redis.set(key, json.dumps(record), keepttl=True)

    with request_context(tenant_id=tenant_id):
        for item in record["items"]:
            # Re-read before each item so a cancellation issued mid-run is seen
            # rather than discovered after the last item.
            current = json.loads(await redis.get(key) or "{}")
            if current.get("status") == "cancelled":
                log.info("batch cancelled mid-run", batch_id=batch_id)
                return {"batch_id": batch_id, "status": "cancelled"}

            result = await _answer_batch_item(item, tenant_id=tenant_id)
            record["results"].append(result)
            record["completed"] += 1
            if result.get("error"):
                record["failed"] += 1
            await redis.set(key, json.dumps(record), keepttl=True)

    record["status"] = "completed"
    record["completed_at"] = _now().isoformat()
    await redis.set(key, json.dumps(record), keepttl=True)

    from src.services.webhooks import dispatch

    await dispatch(
        tenant_id=tenant_id,
        event="batch.completed",
        payload={"id": batch_id, "total": record["total"], "failed": record["failed"]},
    )
    log.info("batch finished", batch_id=batch_id, failed=record["failed"])
    return {"batch_id": batch_id, "status": "completed", "failed": record["failed"]}


async def _answer_batch_item(item: dict[str, Any], *, tenant_id: str) -> dict[str, Any]:
    """Answer one batch item, converting any failure into a recorded result."""
    from src.api.dependencies import build_batch_runtime
    from src.schemas.chat import ChatRequest
    from src.services.chat import ChatService

    try:
        runtime, principal = await build_batch_runtime(tenant_id)
        service = ChatService(runtime=runtime, principal=principal)
        answer = await service.complete(
            ChatRequest(message=item["message"], model=item.get("model"), stream=False)
        )
    except Exception as exc:  # noqa: BLE001 - one item's failure is recorded, not raised
        log.warning("batch item failed", custom_id=item.get("custom_id"), reason=str(exc))
        return {"custom_id": item.get("custom_id"), "error": str(exc)[:500]}

    return {
        "custom_id": item["custom_id"],
        "content": answer.content,
        "model": answer.model,
        "citations": [c.model_dump(mode="json") for c in answer.citations],
        "prompt_tokens": answer.prompt_tokens,
        "completion_tokens": answer.completion_tokens,
    }


@celery_app.task(name="agrag.backup")
def backup() -> dict[str, Any]:
    """Nightly database and index backup."""
    from src.governance.backup import run_backup

    return asyncio.run(run_backup())


@celery_app.task(name="agrag.snapshot_drift")
def snapshot_drift() -> dict[str, Any]:
    """Snapshot retrieval and embedding distributions for drift detection."""
    from src.governance.drift import snapshot_all_tenants

    return asyncio.run(snapshot_all_tenants())


@celery_app.task(name="agrag.recalibrate_judges")
def recalibrate_judges() -> dict[str, Any]:
    """Recompute judge calibration against the human-labelled subset."""
    from src.services.calibration import recalibrate

    return asyncio.run(recalibrate())


@celery_app.task(name="agrag.retry_webhooks")
def retry_webhooks() -> dict[str, Any]:
    """Retry webhook deliveries that are due."""
    from src.services.webhooks import retry_pending

    return asyncio.run(retry_pending())


@celery_app.task(name="agrag.purge_expired")
def purge_expired() -> dict[str, Any]:
    """Delete conversations past their tenant's retention window."""
    from src.governance.retention import purge_expired_conversations

    return asyncio.run(purge_expired_conversations())


def _worker_embedder() -> Any:
    """Build the worker's embedder, falling back when the model is unavailable.

    A worker that cannot embed would fail every job; the hashing fallback keeps
    ingestion moving and is loud about the quality cost.
    """
    from src.ingestion.embedders.base import HashingEmbedder, SentenceTransformerEmbedder

    try:
        return SentenceTransformerEmbedder(settings.default_embedding_model)
    except Exception as exc:  # noqa: BLE001 - degraded, not dead
        log.error(
            "embedding model unavailable in the worker; ingesting with the hashing "
            "fallback, which will produce poor retrieval until this is fixed",
            reason=str(exc),
        )
        return HashingEmbedder(dimension=settings.embedding_dim)


async def _worker_redis() -> Any:
    """Connect to Redis for progress publishing, or return None."""
    try:
        import redis.asyncio as redis_async

        client = redis_async.from_url(settings.redis_url)
        await client.ping()
    except Exception as exc:  # noqa: BLE001 - progress is cosmetic
        log.debug("redis unavailable in the worker", reason=str(exc))
        return None
    return client


async def _worker_sparse() -> Any:
    """Connect to OpenSearch for sparse indexing, or return None."""
    try:
        from opensearchpy import AsyncOpenSearch

        from src.retrieval.sparse import SparseRetriever

        client = AsyncOpenSearch(hosts=[settings.opensearch_url], timeout=15)
        await client.info()
    except Exception as exc:  # noqa: BLE001 - degrade to dense-only
        log.warning("OpenSearch unavailable; documents will be dense-only", reason=str(exc))
        return None
    return SparseRetriever(client=client)


def _now() -> Any:
    """Current UTC time."""
    from datetime import UTC, datetime

    return datetime.now(UTC)
