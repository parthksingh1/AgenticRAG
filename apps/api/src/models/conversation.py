"""Conversation, message, citation and feedback models.

Conversations are stored as a *tree*, not a list: every message carries a
``parent_message_id``, so forking a turn to try a different model or prompt is a
new child rather than a destructive edit. A conversation's visible thread is the
path from the root to ``active_leaf_message_id``.

Example:
    >>> from src.models.conversation import Message, MessageRole
    >>> MessageRole.ASSISTANT.value
    'assistant'
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, JSONColumn, SoftDeleteMixin, TenantScoped, TimestampMixin, new_id


class MessageRole(StrEnum):
    """Author of a message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageStatus(StrEnum):
    """Terminal state of an assistant turn."""

    STREAMING = "streaming"
    COMPLETE = "complete"
    STOPPED = "stopped"
    BLOCKED = "blocked"
    FAILED = "failed"


class FeedbackRating(StrEnum):
    """Thumbs signal from the user. Drives the regression-set growth loop."""

    UP = "up"
    DOWN = "down"


class Conversation(Base, TenantScoped, TimestampMixin, SoftDeleteMixin):
    """A chat thread, possibly branching."""

    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_tenant_id_updated_at", "tenant_id", "updated_at"),
        Index("ix_conversations_tenant_id_user_id", "tenant_id", "user_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("cnv"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64))

    title: Mapped[str] = mapped_column(String(300), default="New conversation", nullable=False)
    is_pinned: Mapped[bool] = mapped_column(
        Boolean, default=False, comment="Pinned demo conversations appear in the sidebar."
    )
    active_leaf_message_id: Mapped[str | None] = mapped_column(
        String(64), comment="Tip of the branch currently displayed."
    )

    # ── Memory ───────────────────────────────────────────────────────────────
    running_summary: Mapped[str | None] = mapped_column(
        Text, comment="LLM-maintained summary of turns that fell out of the window."
    )
    entity_memory: Mapped[dict[str, Any]] = mapped_column(
        JSONColumn,
        default=dict,
        nullable=False,
        comment="Entities extracted across the conversation, keyed by canonical name.",
    )
    summarised_through_ordinal: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    model: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(50))
    conversation_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONColumn, default=dict, nullable=False
    )

    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0, nullable=False)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base, TenantScoped, TimestampMixin):
    """One node in the conversation tree."""

    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
        Index("ix_messages_tenant_id_role", "tenant_id", "role"),
        Index("ix_messages_parent_message_id", "parent_message_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("msg"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    parent_message_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("messages.id", ondelete="CASCADE"),
        comment="Null for the first message. Branching creates siblings.",
    )

    role: Mapped[MessageRole] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    status: Mapped[MessageStatus] = mapped_column(
        String(20), default=MessageStatus.COMPLETE, nullable=False
    )

    # ── Provenance of the answer ─────────────────────────────────────────────
    model: Mapped[str | None] = mapped_column(String(100))
    provider: Mapped[str | None] = mapped_column(String(40))
    prompt_version: Mapped[str | None] = mapped_column(String(50))
    intent: Mapped[str | None] = mapped_column(
        String(40), comment="Label assigned by the intent_router node."
    )
    strategies_used: Mapped[list[str]] = mapped_column(JSONColumn, default=list, nullable=False)
    node_trace: Mapped[list[str]] = mapped_column(
        JSONColumn,
        default=list,
        nullable=False,
        comment="Ordered LangGraph nodes visited, for the thinking panel and replay.",
    )
    tool_calls: Mapped[list[str]] = mapped_column(JSONColumn, default=list, nullable=False)

    # ── Cost and latency ─────────────────────────────────────────────────────
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0, nullable=False)
    ttft_ms: Mapped[int | None] = mapped_column(Integer)
    ttlt_ms: Mapped[int | None] = mapped_column(Integer)

    # ── Quality signals ──────────────────────────────────────────────────────
    cache_hit: Mapped[str | None] = mapped_column(
        String(20), comment="'exact', 'semantic' or null when generated fresh."
    )
    groundedness_score: Mapped[float | None] = mapped_column(Float)
    self_critique: Mapped[str | None] = mapped_column(Text)
    guardrail_flags: Mapped[list[str]] = mapped_column(JSONColumn, default=list, nullable=False)

    trace_id: Mapped[str | None] = mapped_column(String(64), index=True)
    langfuse_trace_url: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    citations: Mapped[list[Citation]] = relationship(
        back_populates="message", cascade="all, delete-orphan", order_by="Citation.marker"
    )
    feedback: Mapped[list[MessageFeedback]] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens for this turn."""
        return self.prompt_tokens + self.completion_tokens


class Citation(Base, TenantScoped, TimestampMixin):
    """Binds a ``[n]`` marker in an answer to the chunk that supports it.

    ``entailment_score`` is produced by the citation_binder node using an NLI
    model. Citations below the tenant's threshold are dropped from the answer
    before it reaches the user, and the drop is recorded here for the eval
    harness to measure citation precision.
    """

    __tablename__ = "citations"
    __table_args__ = (Index("ix_citations_message_id_marker", "message_id", "marker"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("cit"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    message_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("chunks.id", ondelete="SET NULL")
    )
    document_id: Mapped[str | None] = mapped_column(String(64), index=True)

    marker: Mapped[int] = mapped_column(Integer, nullable=False, comment="The n in [n].")
    claim: Mapped[str | None] = mapped_column(
        Text, comment="The sentence in the answer this citation is attached to."
    )
    quote: Mapped[str | None] = mapped_column(Text, comment="Supporting span from the chunk.")

    retrieval_score: Mapped[float | None] = mapped_column(Float)
    rerank_score: Mapped[float | None] = mapped_column(Float)
    entailment_score: Mapped[float | None] = mapped_column(Float)
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    dropped_reason: Mapped[str | None] = mapped_column(String(100))

    message: Mapped[Message] = relationship(back_populates="citations")


class MessageFeedback(Base, TenantScoped, TimestampMixin):
    """A user's thumbs signal, and the triage state that grows the eval set."""

    __tablename__ = "message_feedback"
    __table_args__ = (
        Index("ix_message_feedback_tenant_id_rating", "tenant_id", "rating"),
        Index("ix_message_feedback_promoted", "promoted_to_regression_set"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("fbk"))
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    message_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str | None] = mapped_column(String(64))

    rating: Mapped[FeedbackRating] = mapped_column(String(10), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)

    # Filled in by an admin in the failure explorer.
    failure_mode: Mapped[str | None] = mapped_column(
        String(60),
        comment="e.g. hallucination, missing_citation, wrong_retrieval, refused_wrongly.",
    )
    triaged_by_user_id: Mapped[str | None] = mapped_column(String(64))
    triaged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    promoted_to_regression_set: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )
    regression_case_id: Mapped[str | None] = mapped_column(String(64))

    message: Mapped[Message] = relationship(back_populates="feedback")
