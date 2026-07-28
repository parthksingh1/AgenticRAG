"""Chat endpoints: streaming, non-streaming, history and feedback.

The streaming endpoint is the one that matters. Two properties of it are
deliberate and easy to get wrong:

**The message is persisted before the stream closes, not after.** If the client
disconnects mid-answer — a closed laptop, a flaky connection — the work is
already saved, so reopening the conversation shows the answer rather than losing
it. Persisting after the final token means every disconnect costs a full turn.

**Errors are streamed, not raised.** Once the response has started, the status
code is already sent; raising produces a truncated body with a 200 status, which
the client cannot distinguish from a short answer. An explicit ``error`` event
can be rendered.

Example:
    >>> from src.api.routers.chat import router
    >>> router.prefix
    '/api/chat'
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sse_starlette.sse import EventSourceResponse

from src.api.auth import Principal, get_principal
from src.api.dependencies import TenantRuntime, get_tenant_runtime
from src.core.errors import NotFoundError
from src.core.logging import get_logger
from src.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationOut,
    FeedbackRequest,
    MessageOut,
    StreamEventOut,
)
from src.services.chat import ChatService

log = get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    principal: Annotated[Principal, Depends(get_principal)],
    runtime: Annotated[TenantRuntime, Depends(get_tenant_runtime)],
) -> ChatResponse:
    """Send a message and wait for the complete answer.

    The non-streaming path exists for SDK clients, batch jobs and the eval
    harness, all of which want one object rather than a stream to reassemble.
    """
    service = ChatService(runtime=runtime, principal=principal)
    return await service.complete(request)


@router.post("/stream")
async def stream_message(
    request: ChatRequest,
    principal: Annotated[Principal, Depends(get_principal)],
    runtime: Annotated[TenantRuntime, Depends(get_tenant_runtime)],
) -> EventSourceResponse:
    """Send a message and stream the answer as Server-Sent Events.

    Emits ``start``, then interleaved ``node`` and ``token`` events, then
    ``citations``, then ``done``. The frontend renders ``node`` events as the
    thinking panel and ``token`` events as the answer.
    """
    service = ChatService(runtime=runtime, principal=principal)

    async def events() -> AsyncIterator[str]:
        """Yield SSE frames for one turn."""
        started = time.perf_counter()
        try:
            async for event in service.stream(request):
                yield event.to_sse()
        except Exception as exc:
            # Raising here would truncate the body under an already-sent 200,
            # which the client cannot tell apart from a short answer.
            log.exception("streaming turn failed", error=str(exc))
            yield StreamEventOut(type="error", error="The answer could not be completed.").to_sse()
        finally:
            log.info(
                "chat turn finished",
                duration_ms=int((time.perf_counter() - started) * 1000),
                conversation_id=request.conversation_id,
            )

    return EventSourceResponse(
        events(),
        headers={
            # Nginx and several CDNs buffer by default, which turns a token
            # stream into one delivery at the end and defeats the whole feature.
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
        },
    )


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    principal: Annotated[Principal, Depends(get_principal)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ConversationOut]:
    """List the workspace's conversations, most recently updated first."""
    from src.core.db import session_scope
    from src.repositories.conversations import list_conversations as query

    async with session_scope() as session:
        return await query(session, tenant_id=principal.tenant_id, limit=limit, offset=offset)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    branch_from: Annotated[str | None, Query()] = None,
) -> ConversationDetail:
    """Load a conversation and the messages on its active branch.

    ``branch_from`` selects a different leaf, which is how the UI switches
    between forks without mutating anything.
    """
    from src.core.db import session_scope
    from src.repositories.conversations import load_conversation

    async with session_scope() as session:
        conversation = await load_conversation(
            session,
            conversation_id=conversation_id,
            tenant_id=principal.tenant_id,
            leaf_message_id=branch_from,
        )

    if conversation is None:
        raise NotFoundError("Conversation not found.")
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
) -> None:
    """Soft-delete a conversation.

    Soft rather than hard, so an accidental delete is recoverable within the
    retention window. Erasure under GDPR is a separate, genuinely destructive
    path in :mod:`src.governance.gdpr`.
    """
    from src.core.db import session_scope
    from src.repositories.conversations import soft_delete_conversation

    async with session_scope() as session:
        deleted = await soft_delete_conversation(
            session, conversation_id=conversation_id, tenant_id=principal.tenant_id
        )

    if not deleted:
        raise NotFoundError("Conversation not found.")


@router.get("/messages/{message_id}", response_model=MessageOut)
async def get_message(
    message_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
) -> MessageOut:
    """Load a single message with its citations."""
    from src.core.db import session_scope
    from src.repositories.conversations import load_message

    async with session_scope() as session:
        message = await load_message(session, message_id=message_id, tenant_id=principal.tenant_id)

    if message is None:
        raise NotFoundError("Message not found.")
    return message


@router.post("/messages/{message_id}/feedback", status_code=204)
async def submit_feedback(
    message_id: str,
    request: FeedbackRequest,
    principal: Annotated[Principal, Depends(get_principal)],
) -> None:
    """Record a thumbs signal.

    A thumbs-down is the first step of the loop that grows the regression set:
    it lands in the failure explorer, an admin labels the failure mode, and the
    case is promoted with an expected answer.
    """
    from src.core.db import session_scope
    from src.repositories.conversations import record_feedback

    async with session_scope() as session:
        recorded = await record_feedback(
            session,
            message_id=message_id,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            rating=request.rating,
            comment=request.comment,
        )

    if not recorded:
        raise NotFoundError("Message not found.")


@router.post("/messages/{message_id}/regenerate", response_model=ChatResponse)
async def regenerate(
    message_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    runtime: Annotated[TenantRuntime, Depends(get_tenant_runtime)],
    model: Annotated[str | None, Query()] = None,
) -> ChatResponse:
    """Regenerate an answer as a sibling branch.

    The original is kept. Overwriting it would lose the comparison the user
    asked for by regenerating, and would silently discard a message someone may
    have already cited elsewhere.
    """
    service = ChatService(runtime=runtime, principal=principal)
    return await service.regenerate(message_id=message_id, model=model)
