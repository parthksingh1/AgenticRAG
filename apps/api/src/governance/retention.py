"""Data retention.

Tenants set a retention window; conversations past it are deleted. Two details
matter more than the sweep itself.

**Pinned conversations survive.** A pinned conversation is one someone deliberately
kept, and a retention policy that silently deletes the thing a user explicitly
saved reads as data loss regardless of what the policy said.

**Deletion is hard, not soft.** A retention policy that only sets ``deleted_at``
does not satisfy the promise it makes — the data is still there, still
recoverable, still in the backup. Soft delete is for user-initiated deletes,
where undo is a feature.

Example:
    >>> cutoff_for(30, now=datetime(2026, 3, 31, tzinfo=UTC)).date().isoformat()
    '2026-03-01'
    >>> cutoff_for(None) is None
    True
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)

#: Conversations deleted per tenant per run. A tenant switching from unlimited
#: retention to 30 days would otherwise delete millions of rows in one
#: transaction and hold locks long enough to stall live traffic.
BATCH_LIMIT = 5_000


def cutoff_for(retention_days: int | None, *, now: datetime | None = None) -> datetime | None:
    """The timestamp before which data is expired, or None for unlimited.

    Example:
        >>> cutoff_for(1, now=datetime(2026, 1, 2, tzinfo=UTC)).isoformat()
        '2026-01-01T00:00:00+00:00'
    """
    if retention_days is None:
        return None
    return (now or datetime.now(UTC)) - timedelta(days=retention_days)


async def purge_expired_conversations() -> dict[str, Any]:
    """Delete conversations past each tenant's retention window.

    Runs daily. Returns per-tenant counts so the admin UI can show what the
    policy actually did rather than only what it promises.
    """
    from sqlalchemy import delete, select

    from src.core.context import request_context
    from src.core.db import session_scope, system_session
    from src.models.conversation import Citation, Conversation, Message, MessageFeedback
    from src.models.tenant import Tenant

    async with system_session(reason="daily retention purge") as session:
        policies = [
            (row[0], row[1])
            for row in (
                await session.execute(
                    select(Tenant.id, Tenant.retention_days).where(
                        Tenant.deleted_at.is_(None), Tenant.retention_days.is_not(None)
                    )
                )
            ).all()
        ]

    purged: dict[str, int] = {}
    for tenant_id, retention_days in policies:
        cutoff = cutoff_for(retention_days)
        if cutoff is None:  # pragma: no cover - filtered by the query above
            continue

        with request_context(tenant_id=tenant_id):
            async with session_scope() as session:
                ids = [
                    row[0]
                    for row in (
                        await session.execute(
                            select(Conversation.id)
                            .where(
                                Conversation.created_at < cutoff,
                                # A pinned conversation was deliberately kept.
                                Conversation.is_pinned.is_(False),
                            )
                            .limit(BATCH_LIMIT)
                        )
                    ).all()
                ]
                if not ids:
                    continue

                message_ids = [
                    row[0]
                    for row in (
                        await session.execute(
                            select(Message.id).where(Message.conversation_id.in_(ids))
                        )
                    ).all()
                ]

                # Children first: the foreign keys are not ON DELETE CASCADE for
                # citations, and relying on cascade behaviour that differs
                # between Postgres and SQLite would make the tests lie.
                if message_ids:
                    await session.execute(
                        delete(Citation).where(Citation.message_id.in_(message_ids))
                    )
                    await session.execute(
                        delete(MessageFeedback).where(MessageFeedback.message_id.in_(message_ids))
                    )
                await session.execute(delete(Message).where(Message.conversation_id.in_(ids)))
                await session.execute(delete(Conversation).where(Conversation.id.in_(ids)))

        purged[tenant_id] = len(ids)
        log.info(
            "purged conversations past the retention window",
            tenant_id=tenant_id,
            retention_days=retention_days,
            deleted=len(ids),
        )

    return {"tenants": len(policies), "purged": purged, "total": sum(purged.values())}
