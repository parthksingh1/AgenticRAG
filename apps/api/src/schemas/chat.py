"""Chat API request and response schemas.

Separate from the ORM models on purpose. A response schema that is just the ORM
model leaks every column the moment someone adds one — including the internal
scores, the tenant id and the trace ids — and couples the public contract to the
database shape, so a migration becomes a breaking API change.

Example:
    >>> ChatRequest(message="hello").stream
    True
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.models.conversation import FeedbackRating, MessageRole


class CitationOut(BaseModel):
    """A citation as the frontend renders it."""

    model_config = ConfigDict(frozen=True)

    marker: int = Field(description="The n in [n], 1-based.")
    chunk_id: str | None = None
    document_id: str | None = None
    document_title: str | None = None
    page_number: int | None = None
    section: str | None = None
    quote: str | None = Field(default=None, description="The supporting span, for highlighting.")
    verified: bool = Field(
        default=False,
        description="Whether an NLI check confirmed the cited passage supports the claim.",
    )
    score: float | None = None


class ChatRequest(BaseModel):
    """A user's turn."""

    model_config = ConfigDict(frozen=True)

    message: str = Field(min_length=1, max_length=32_000)
    conversation_id: str | None = Field(
        default=None, description="Omit to start a new conversation."
    )
    parent_message_id: str | None = Field(
        default=None,
        description="Fork from this message instead of appending to the current branch.",
    )
    model: str | None = Field(default=None, description="Overrides the workspace default.")
    prompt_version: str | None = None
    stream: bool = True
    #: Restrict retrieval to specific documents, for "ask this document" flows.
    document_ids: tuple[str, ...] = ()
    include_thinking: bool = Field(
        default=True, description="Emit node-level progress events while streaming."
    )


class ChatResponse(BaseModel):
    """A completed assistant turn."""

    model_config = ConfigDict(frozen=True)

    message_id: str
    conversation_id: str
    content: str
    citations: tuple[CitationOut, ...] = ()
    model: str
    intent: str | None = None
    #: Ordered graph nodes visited, so the UI can replay the thinking panel for a
    #: message loaded from history rather than only for a live stream.
    node_trace: tuple[str, ...] = ()
    strategies_used: tuple[str, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    ttft_ms: int | None = None
    ttlt_ms: int | None = None
    cache_hit: str | None = None
    groundedness: float | None = None
    guardrail_flags: tuple[str, ...] = ()
    stop_reason: str | None = None
    trace_id: str | None = None
    langfuse_url: str | None = None


class MessageOut(BaseModel):
    """A message as returned when loading a conversation."""

    model_config = ConfigDict(frozen=True)

    id: str
    role: MessageRole
    content: str
    created_at: datetime
    parent_message_id: str | None = None
    citations: tuple[CitationOut, ...] = ()
    model: str | None = None
    intent: str | None = None
    node_trace: tuple[str, ...] = ()
    cost_usd: float = 0.0
    ttft_ms: int | None = None
    cache_hit: str | None = None
    feedback: str | None = None
    #: Sibling count, so the UI can render "2 of 3" branch navigation without a
    #: second request.
    branch_count: int = 1


class ConversationOut(BaseModel):
    """A conversation summary for the sidebar."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    is_pinned: bool = False
    total_cost_usd: float = 0.0
    model: str | None = None


class ConversationDetail(ConversationOut):
    """A conversation with its currently displayed branch."""

    messages: tuple[MessageOut, ...] = ()
    running_summary: str | None = None


class FeedbackRequest(BaseModel):
    """A thumbs signal on an assistant message."""

    model_config = ConfigDict(frozen=True)

    rating: FeedbackRating
    comment: str | None = Field(default=None, max_length=4000)


class StreamEventOut(BaseModel):
    """One server-sent event in a streaming turn.

    Deliberately a single flat type rather than a union: the frontend switches on
    ``type``, and a discriminated union across a dozen event shapes would make
    both the SDK and the TypeScript client harder to use for no benefit.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["start", "node", "token", "citations", "done", "error", "usage", "tool"]
    #: Set on ``token`` events.
    text: str | None = None
    #: Set on ``node`` events: which graph node, and what it produced.
    node: str | None = None
    status: str | None = None
    detail: str | None = None
    data: dict[str, Any] | None = None
    #: Set on the ``done`` event.
    message_id: str | None = None
    conversation_id: str | None = None
    citations: tuple[CitationOut, ...] | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None

    def to_sse(self) -> str:
        """Render as a Server-Sent Event frame.

        Example:
            >>> StreamEventOut(type="token", text="hi").to_sse().startswith("event: token")
            True
        """
        import json

        payload = self.model_dump(exclude_none=True, mode="json")
        return f"event: {self.type}\ndata: {json.dumps(payload)}\n\n"
