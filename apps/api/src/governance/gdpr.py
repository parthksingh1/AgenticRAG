"""Right to erasure and right to export.

A tenant's data lives in five places — Postgres, MinIO, OpenSearch, Neo4j and
Redis — and "delete my data" means all five. A cascade that covers only the
database leaves the document bytes in object storage, the text in the BM25
index, the extracted entities in the graph and the cached answers in Redis, any
one of which makes the deletion a lie.

Erasure is therefore explicit per store, and the result reports what each store
actually removed. A partial erasure is reported as partial, not rounded up to
success: telling a regulator "deleted" when OpenSearch was unreachable is worse
than telling them "deleted from four of five stores, retry scheduled".

Deletion order matters. Postgres goes **last**, because its rows are the index
of what to delete everywhere else; losing them first would strand the rest with
no way to find it.

Example:
    >>> ErasureReport(tenant_id="t").partial
    False
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)

#: Tables erased for a tenant, in dependency order (children before parents).
#: Written out rather than derived from ``Base.metadata`` on purpose: a new table
#: must be a deliberate decision here, and a derived list would silently start
#: deleting from a table nobody considered.
ERASURE_ORDER = (
    "citations",
    "message_feedback",
    "messages",
    "conversations",
    "chunks",
    "document_versions",
    "ingestion_jobs",
    "documents",
    "retrieval_logs",
    "guardrail_events",
    "usage_records",
    "tenant_budget_counters",
    "webhook_deliveries",
    "webhook_endpoints",
    "experiments",
    "drift_snapshots",
    "api_keys",
    "users",
    "audit_logs",
)


@dataclass(slots=True)
class ErasureReport:
    """What each store reported after an erasure."""

    tenant_id: str
    stores: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def partial(self) -> bool:
        """Whether any store failed.

        Example:
            >>> r = ErasureReport(tenant_id="t")
            >>> r.errors["opensearch"] = "connection refused"
            >>> r.partial
            True
        """
        return bool(self.errors)

    def as_dict(self) -> dict[str, Any]:
        """Render the report for the API and the audit log."""
        return {
            "tenant_id": self.tenant_id,
            "stores": self.stores,
            "errors": self.errors,
            "complete": not self.partial,
            "started_at": self.started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
        }


async def erase_tenant(tenant_id: str, *, dry_run: bool = False) -> dict[str, Any]:
    """Erase every trace of a tenant across all five stores.

    Args:
        tenant_id: The tenant to erase.
        dry_run: Count what would be deleted without deleting it. The default is
            False, but every UI path should offer the dry run first — this is the
            one operation in the system with no undo.

    Returns:
        A per-store report. Check ``complete`` before telling anyone the data is
        gone.
    """
    from src.governance import audit

    report = ErasureReport(tenant_id=tenant_id)

    # Object storage, search index, graph and cache first; Postgres last,
    # because its rows are the map of what to erase elsewhere.
    for name, eraser in (
        ("object_storage", _erase_objects),
        ("opensearch", _erase_sparse),
        ("neo4j", _erase_graph),
        ("redis", _erase_cache),
        ("postgres", _erase_rows),
    ):
        try:
            report.stores[name] = await eraser(tenant_id, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001 - one store failing must not stop the rest
            report.errors[name] = str(exc)[:500]
            log.error("erasure failed for a store", store=name, tenant_id=tenant_id, error=str(exc))

    if not dry_run:
        await audit.record(
            action="gdpr.erasure",
            tenant_id=tenant_id,
            resource_type="tenant",
            resource_id=tenant_id,
            after=report.as_dict(),
            success=not report.partial,
        )

    if report.partial:
        log.error(
            "tenant erasure completed only partially; the remaining stores must be retried",
            tenant_id=tenant_id,
            failed=sorted(report.errors),
        )
    return report.as_dict()


async def export_tenant(tenant_id: str) -> dict[str, Any]:
    """Export everything held about a tenant, for the right to data portability.

    Returned as JSON-serialisable structures rather than a file: the caller
    decides whether that becomes a download, an object in storage or a stream.
    Documents are exported as metadata plus storage keys; the bytes are fetched
    through presigned URLs so a 2GB corpus is not marshalled through this
    process.
    """
    from sqlalchemy import select

    from src.core.context import request_context
    from src.core.db import session_scope
    from src.governance import audit
    from src.models.conversation import Conversation, Message
    from src.models.document import Document
    from src.models.telemetry import AuditLog, UsageRecord
    from src.models.tenant import Tenant, User

    with request_context(tenant_id=tenant_id):
        async with session_scope() as session:
            tenant = await session.get(Tenant, tenant_id)
            documents = (await session.execute(select(Document))).scalars().all()
            conversations = (await session.execute(select(Conversation))).scalars().all()
            messages = (await session.execute(select(Message))).scalars().all()
            users = (await session.execute(select(User))).scalars().all()
            usage = (await session.execute(select(UsageRecord).limit(50_000))).scalars().all()
            audits = (await session.execute(select(AuditLog).limit(50_000))).scalars().all()

    payload = {
        "exported_at": datetime.now(UTC).isoformat(),
        "format_version": "1.0",
        "tenant": _row(tenant) if tenant else None,
        "users": [_row(u) for u in users],
        "documents": [_row(d) for d in documents],
        "conversations": [_row(c) for c in conversations],
        "messages": [_row(m) for m in messages],
        "usage_records": [_row(u) for u in usage],
        "audit_log": [_row(a) for a in audits],
    }

    await audit.record(
        action="gdpr.export",
        tenant_id=tenant_id,
        resource_type="tenant",
        resource_id=tenant_id,
        after={"documents": len(documents), "messages": len(messages)},
    )
    return payload


async def _erase_rows(tenant_id: str, *, dry_run: bool) -> dict[str, int]:
    """Delete the tenant's rows from Postgres, children first.

    Runs raw deletes in one transaction rather than ORM cascades: loading a
    million chunks into the identity map to delete them would exhaust memory,
    and a partial cascade would leave orphans that no later run can find.
    """
    from sqlalchemy import text

    from src.core.db import system_session

    counts: dict[str, int] = {}
    async with system_session(reason="GDPR erasure") as session:
        for table in ERASURE_ORDER:
            verb = "SELECT count(*) FROM" if dry_run else "DELETE FROM"
            # The table name comes from ERASURE_ORDER, a module constant, never
            # from a request; the tenant id is bound as a parameter.
            statement = text(f"{verb} {table} WHERE tenant_id = :tenant_id")
            result = await session.execute(statement, {"tenant_id": tenant_id})
            counts[table] = int(result.scalar_one()) if dry_run else int(result.rowcount or 0)

        if not dry_run:
            await session.execute(
                text("DELETE FROM tenants WHERE id = :tenant_id"), {"tenant_id": tenant_id}
            )
            counts["tenants"] = 1

    return counts


async def _erase_objects(tenant_id: str, *, dry_run: bool) -> dict[str, int]:
    """Delete the tenant's objects from MinIO."""
    import asyncio

    from src.core.config import get_settings
    from src.services.storage import ObjectStorage

    storage = ObjectStorage(get_settings())
    if dry_run:
        return {"objects": await asyncio.to_thread(storage.count_tenant_objects, tenant_id)}
    return {"objects": await asyncio.to_thread(storage.delete_tenant, tenant_id)}


async def _erase_sparse(tenant_id: str, *, dry_run: bool) -> dict[str, int]:
    """Delete the tenant's documents from the BM25 index."""
    from opensearchpy import AsyncOpenSearch

    from src.core.config import get_settings

    settings = get_settings()
    client = AsyncOpenSearch(hosts=[settings.opensearch_url], timeout=30)
    query = {"query": {"term": {"tenant_id": tenant_id}}}
    try:
        if dry_run:
            result = await client.count(index=settings.opensearch_index, body=query)
            return {"documents": int(result.get("count", 0))}
        result = await client.delete_by_query(
            index=settings.opensearch_index, body=query, refresh=True
        )
        return {"documents": int(result.get("deleted", 0))}
    finally:
        await client.close()


async def _erase_graph(tenant_id: str, *, dry_run: bool) -> dict[str, int]:
    """Delete the tenant's nodes and relationships from Neo4j.

    ``DETACH DELETE`` removes the relationships with the nodes; a plain DELETE
    fails on any node that still has one, which is every node worth having.
    """
    from neo4j import AsyncGraphDatabase

    from src.core.config import get_settings

    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_url, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    cypher = (
        "MATCH (n {tenant_id: $tenant_id}) RETURN count(n) AS n"
        if dry_run
        else "MATCH (n {tenant_id: $tenant_id}) DETACH DELETE n RETURN count(n) AS n"
    )
    try:
        async with driver.session() as session:
            result = await session.run(cypher, tenant_id=tenant_id)
            record = await result.single()
            return {"nodes": int(record["n"]) if record else 0}
    finally:
        await driver.close()


async def _erase_cache(tenant_id: str, *, dry_run: bool) -> dict[str, int]:
    """Delete the tenant's cached answers, embeddings and counters from Redis.

    Uses ``scan_iter`` rather than ``KEYS``: ``KEYS`` blocks the whole server
    while it walks the keyspace, and a cache that stalls during a deletion takes
    every request with it.
    """
    import redis.asyncio as redis_async

    from src.core.config import get_settings

    client = redis_async.from_url(get_settings().redis_url)
    deleted = 0
    try:
        async for key in client.scan_iter(match=f"*:{tenant_id}:*", count=500):
            deleted += 1
            if not dry_run:
                await client.delete(key)
    finally:
        await client.aclose()
    return {"keys": deleted}


def _row(instance: Any) -> dict[str, Any]:
    """Render an ORM row as JSON-safe primitives.

    Example:
        >>> class Fake:
        ...     __table__ = None
        >>> _row(None)
        {}
    """
    if instance is None or getattr(instance, "__table__", None) is None:
        return {}
    out: dict[str, Any] = {}
    for column in instance.__table__.columns:
        value = getattr(instance, column.name, None)
        out[column.name] = json.loads(json.dumps(value, default=str))
    return out
