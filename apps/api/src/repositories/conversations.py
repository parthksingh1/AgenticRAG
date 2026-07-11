"""Conversation, message and citation persistence.

Conversations are a tree. The one query that matters is "give me the path from
the root to a leaf", because that is what the UI renders, and doing it naively —
loading every message and walking in Python — is fine at ten messages and
untenable at a thousand.

It is done as a recursive CTE: one round trip, and the database walks the parent
chain. The alternative of storing a materialised path column would be faster
still but has to be maintained on every branch operation, which is exactly where
a bug would silently corrupt history.

Example:
    >>> from src.repositories.conversations import DEFAULT_TITLE_LENGTH
    >>> DEFAULT_TITLE_LENGTH
    60
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.logging import get_logger
from src.models.conversation import (
    Citation,
    Conversation,
    FeedbackRating,
    Message,
    MessageFeedback,
    MessageRole,
)
from src.schemas.chat import CitationOut, ConversationDetail, ConversationOut, MessageOut

log = get_logger(__name__)

#: Conversations are titled from the first question, truncated to this.
DEFAULT_TITLE_LENGTH = 60

#: Walks from a leaf to the root by following parent_message_id, then reverses.
#: Bounded by depth so a cycle — which should be impossible, but would otherwise
#: hang the query forever — terminates.
_BRANCH_PATH_SQL = text(
    """
    WITH RECURSIVE branch AS (
        SELECT m.id, m.parent_message_id, 0 AS depth
        FROM messages m
        WHERE m.id = :leaf_id AND m.tenant_id = :tenant_id
        UNION ALL
        SELECT p.id, p.parent_message_id, branch.depth + 1
        FROM messages p
        JOIN branch ON p.id = branch.parent_message_id
        WHERE p.tenant_id = :tenant_id AND branch.depth < 500
    )
    SELECT id FROM branch ORDER BY depth DESC
    """
)


def title_from(question: str) -> str:
    """Derive a conversation title from its first message.

    Example:
        >>> title_from("What is our refund policy?")
        'What is our refund policy?'
        >>> len(title_from("x" * 200))
        63
    """
    cleaned = " ".join(question.split())
    if len(cleaned) <= DEFAULT_TITLE_LENGTH:
        return cleaned or "New conversation"
    return cleaned[:DEFAULT_TITLE_LENGTH] + "..."


async def get_or_create_conversation(
    session: AsyncSession,
    *,
    conversation_id: str | None,
    tenant_id: str,
    user_id: str | None,
    first_message: str,
    model: str | None = None,
) -> Conversation:
    """Load a conversation, or start one titled from the first question."""
    if conversation_id:
        result = await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        existing = result.scalar_one_or_none()
        if existing is not None:
            return existing
        log.info("conversation not found; starting a new one", requested=conversation_id)

    conversation = Conversation(
        tenant_id=tenant_id,
        user_id=user_id,
        title=title_from(first_message),
        model=model,
    )
    session.add(conversation)
    await session.flush()
    return conversation


async def append_message(
    session: AsyncSession,
    *,
    conversation: Conversation,
    role: MessageRole,
    content: str,
    parent_message_id: str | None = None,
    **fields: Any,
) -> Message:
    """Append a message to a conversation branch.

    Parenting to the conversation's current leaf by default is what makes the
    common case a list; passing ``parent_message_id`` explicitly is what makes a
    branch.
    """
    message = Message(
        tenant_id=conversation.tenant_id,
        conversation_id=conversation.id,
        parent_message_id=parent_message_id or conversation.active_leaf_message_id,
        role=role,
        content=content,
        **fields,
    )
    session.add(message)
    await session.flush()

    conversation.active_leaf_message_id = message.id
    conversation.updated_at = datetime.now(UTC)
    return message


async def save_citations(
    session: AsyncSession, *, message: Message, citations: list[dict[str, Any]]
) -> None:
    """Persist a message's verified citations.

    Unverified citations are stored too, with ``verified=False`` and the reason
    they were dropped: the eval harness measures citation precision from exactly
    this, and keeping only the survivors would make the metric unmeasurable.
    """
    for citation in citations:
        session.add(
            Citation(
                tenant_id=message.tenant_id,
                message_id=message.id,
                chunk_id=citation.get("chunk_id"),
                document_id=citation.get("document_id"),
                marker=int(citation.get("marker", 0)),
                claim=citation.get("claim"),
                quote=citation.get("quote"),
                retrieval_score=citation.get("retrieval_score"),
                rerank_score=citation.get("rerank_score"),
                entailment_score=citation.get("score"),
                verified=bool(citation.get("verified")),
                dropped_reason=citation.get("entailment") if not citation.get("verified") else None,
            )
        )


async def list_conversations(
    session: AsyncSession, *, tenant_id: str, limit: int = 50, offset: int = 0
) -> list[ConversationOut]:
    """List conversations with their message counts.

    The count comes from a correlated subquery rather than loading the messages:
    the sidebar needs a number, not a thousand rows.
    """
    message_count = (
        select(func.count(Message.id))
        .where(Message.conversation_id == Conversation.id)
        .correlate(Conversation)
        .scalar_subquery()
    )

    result = await session.execute(
        select(Conversation, message_count.label("message_count"))
        .where(Conversation.deleted_at.is_(None))
        .order_by(Conversation.is_pinned.desc(), Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )

    return [
        ConversationOut(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=int(count or 0),
            is_pinned=conversation.is_pinned,
            total_cost_usd=float(conversation.total_cost_usd or 0),
            model=conversation.model,
        )
        for conversation, count in result.all()
    ]


async def load_conversation(
    session: AsyncSession,
    *,
    conversation_id: str,
    tenant_id: str,
    leaf_message_id: str | None = None,
) -> ConversationDetail | None:
    """Load a conversation and the messages along one branch."""
    result = await session.execute(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.deleted_at.is_(None)
        )
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        return None

    leaf = leaf_message_id or conversation.active_leaf_message_id
    messages = await _load_branch(session, leaf=leaf, tenant_id=tenant_id)
    siblings = await _sibling_counts(session, conversation_id=conversation_id)

    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=len(messages),
        is_pinned=conversation.is_pinned,
        total_cost_usd=float(conversation.total_cost_usd or 0),
        model=conversation.model,
        running_summary=conversation.running_summary,
        messages=tuple(
            _to_message_out(m, branch_count=siblings.get(m.parent_message_id or "", 1))
            for m in messages
        ),
    )


async def _load_branch(session: AsyncSession, *, leaf: str | None, tenant_id: str) -> list[Message]:
    """Load the messages from the root to a leaf, in order."""
    if not leaf:
        return []

    ordered = await session.execute(_BRANCH_PATH_SQL, {"leaf_id": leaf, "tenant_id": tenant_id})
    ids = [row[0] for row in ordered.all()]
    if not ids:
        return []

    result = await session.execute(
        select(Message).where(Message.id.in_(ids)).options(selectinload(Message.citations))
    )
    by_id = {m.id: m for m in result.scalars().all()}
    return [by_id[i] for i in ids if i in by_id]


async def _sibling_counts(session: AsyncSession, *, conversation_id: str) -> dict[str, int]:
    """How many children each message has, for the branch navigator.

    One grouped query rather than a count per message: the alternative is N+1
    queries on every conversation load.
    """
    result = await session.execute(
        select(Message.parent_message_id, func.count(Message.id))
        .where(Message.conversation_id == conversation_id, Message.parent_message_id.isnot(None))
        .group_by(Message.parent_message_id)
    )
    return {parent: count for parent, count in result.all() if parent}


def _to_message_out(message: Message, *, branch_count: int = 1) -> MessageOut:
    """Map an ORM message onto its API shape."""
    return MessageOut(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        parent_message_id=message.parent_message_id,
        citations=tuple(
            CitationOut(
                marker=c.marker,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                quote=c.quote,
                verified=c.verified,
                score=c.entailment_score,
            )
            for c in sorted(message.citations, key=lambda c: c.marker)
        ),
        model=message.model,
        intent=message.intent,
        node_trace=tuple(message.node_trace or ()),
        cost_usd=float(message.cost_usd or 0),
        ttft_ms=message.ttft_ms,
        cache_hit=message.cache_hit,
        branch_count=branch_count,
    )


async def load_message(
    session: AsyncSession, *, message_id: str, tenant_id: str
) -> MessageOut | None:
    """Load one message with its citations."""
    result = await session.execute(
        select(Message).where(Message.id == message_id).options(selectinload(Message.citations))
    )
    message = result.scalar_one_or_none()
    return _to_message_out(message) if message is not None else None


async def history_for(
    session: AsyncSession, *, conversation: Conversation, window: int = 10
) -> list[Message]:
    """The last ``window`` messages on the active branch, oldest first.

    A window rather than the whole conversation: turns beyond it are represented
    by the running summary, which is what keeps a long conversation's prompt from
    growing without bound.
    """
    branch = await _load_branch(
        session, leaf=conversation.active_leaf_message_id, tenant_id=conversation.tenant_id
    )
    return branch[-window:]


async def soft_delete_conversation(
    session: AsyncSession, *, conversation_id: str, tenant_id: str
) -> bool:
    """Mark a conversation deleted, returning whether it existed."""
    result = await session.execute(select(Conversation).where(Conversation.id == conversation_id))
    conversation = result.scalar_one_or_none()
    if conversation is None:
        return False

    conversation.deleted_at = datetime.now(UTC)
    return True


async def record_feedback(
    session: AsyncSession,
    *,
    message_id: str,
    tenant_id: str,
    user_id: str | None,
    rating: FeedbackRating,
    comment: str | None,
) -> bool:
    """Record or replace a user's thumbs signal on a message.

    Replaces rather than appends: a user changing their mind should leave one
    signal, not two contradictory ones that both feed the failure explorer.
    """
    result = await session.execute(select(Message).where(Message.id == message_id))
    if result.scalar_one_or_none() is None:
        return False

    existing = await session.execute(
        select(MessageFeedback).where(
            MessageFeedback.message_id == message_id,
            MessageFeedback.user_id == user_id,
        )
    )
    feedback = existing.scalar_one_or_none()

    if feedback is not None:
        feedback.rating = rating
        feedback.comment = comment
        return True

    session.add(
        MessageFeedback(
            tenant_id=tenant_id,
            message_id=message_id,
            user_id=user_id,
            rating=rating,
            comment=comment,
        )
    )
    return True


async def update_memory(
    session: AsyncSession,
    *,
    conversation: Conversation,
    summary: str | None,
    entities: dict[str, Any] | None,
    through_ordinal: int,
) -> None:
    """Persist the conversation's running summary and entity memory."""
    if summary is not None:
        conversation.running_summary = summary
    if entities is not None:
        conversation.entity_memory = {**(conversation.entity_memory or {}), **entities}
    conversation.summarised_through_ordinal = through_ordinal
