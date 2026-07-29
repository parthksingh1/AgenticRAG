"""The OpenAI-compatible endpoint.

``POST /v1/chat/completions`` accepts and returns the OpenAI shape, so any
OpenAI SDK works against this system with only a ``base_url`` change:

```python
client = OpenAI(base_url="https://api.example.com/v1", api_key="agr_...")
client.chat.completions.create(model="claude-sonnet-5", messages=[...])
```

That compatibility is the difference between an API someone can try in two
minutes and one they have to read about first.

Two deliberate divergences:

* **Citations are added** as a non-standard ``citations`` field. An OpenAI-shaped
  RAG response with no citations is indistinguishable from a plain completion,
  which throws away the only thing that makes this system worth pointing at.
  Clients that ignore unknown fields are unaffected.
* **Unsupported parameters are accepted and ignored**, not rejected. A client
  sending ``frequency_penalty`` should get an answer, not a 400 about a
  parameter that does not apply to a retrieval pipeline.

Example:
    >>> from src.api.routers.openai_compat import router
    >>> router.prefix
    '/v1'
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from src.api.auth import Principal, get_principal
from src.api.dependencies import TenantRuntime, get_tenant_runtime
from src.core.errors import ValidationFailedError
from src.core.logging import get_logger
from src.schemas.admin import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChoice,
    OpenAIMessage,
    OpenAIUsage,
)
from src.schemas.chat import ChatRequest
from src.services.chat import ChatService

log = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compatible"])


@router.post(
    "/chat/completions",
    # response_model=None because the return type is a union of a model and a
    # streaming response, which FastAPI cannot turn into one schema. The 200
    # shape is declared explicitly instead, so the OpenAPI document — and every
    # SDK generated from it — still knows what a non-streaming answer looks like.
    response_model=None,
    responses={200: {"model": OpenAIChatResponse, "description": "A chat completion."}},
)
async def chat_completions(
    request: OpenAIChatRequest,
    principal: Annotated[Principal, Depends(get_principal)],
    runtime: Annotated[TenantRuntime, Depends(get_tenant_runtime)],
) -> OpenAIChatResponse | EventSourceResponse:
    """Answer in the OpenAI chat-completions shape."""
    question = _last_user_message(request)
    service = ChatService(runtime=runtime, principal=principal)

    internal = ChatRequest(
        message=question,
        model=request.model,
        stream=request.stream,
        include_thinking=False,
    )

    if request.stream:
        return EventSourceResponse(
            _stream_chunks(service, internal, model=request.model),
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    answer = await service.complete(internal)
    return OpenAIChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=answer.model,
        choices=(
            OpenAIChoice(
                index=0,
                message=OpenAIMessage(role="assistant", content=answer.content),
                finish_reason=_finish_reason(answer.stop_reason),
            ),
        ),
        usage=OpenAIUsage(
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
            total_tokens=answer.prompt_tokens + answer.completion_tokens,
        ),
        citations=tuple(c.model_dump(mode="json") for c in answer.citations),
    )


async def _stream_chunks(
    service: ChatService, request: ChatRequest, *, model: str
) -> AsyncIterator[str]:
    """Emit OpenAI-shaped streaming chunks.

    The wire format is exactly what the OpenAI SDK expects, including the
    ``[DONE]`` sentinel — without it the SDK's iterator hangs waiting for a
    terminator that never arrives.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta: dict, finish: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant", "content": ""})

    citations: tuple = ()
    try:
        async for event in service.stream(request):
            if event.type == "token" and event.text:
                yield chunk({"content": event.text})
            elif event.type == "citations" and event.citations:
                citations = event.citations
            elif event.type == "error":
                yield chunk({}, finish="error")
                yield "data: [DONE]\n\n"
                return
    except Exception as exc:
        log.exception("OpenAI-compatible stream failed", error=str(exc))
        yield chunk({}, finish="error")
        yield "data: [DONE]\n\n"
        return

    if citations:
        # Sent as a final delta so a standard client sees a normal completion and
        # a client that cares can read the citations off the last chunk.
        yield (
            "data: "
            + json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                    "citations": [c.model_dump(mode="json") for c in citations],
                }
            )
            + "\n\n"
        )

    yield chunk({}, finish="stop")
    yield "data: [DONE]\n\n"


@router.get("/models")
async def list_models(
    principal: Annotated[Principal, Depends(get_principal)],
    runtime: Annotated[TenantRuntime, Depends(get_tenant_runtime)],
) -> dict:
    """List the models this workspace may use, in the OpenAI shape.

    SDK clients call this to populate a model picker, and returning the
    workspace's allowlist rather than every known model means the picker cannot
    offer something the policy will then refuse.
    """
    from src.services.llm.pricing import MODEL_PROVIDERS

    policy = runtime.router._policy
    allowed = policy.allowed_models or tuple(MODEL_PROVIDERS)
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": MODEL_PROVIDERS.get(model, "unknown"),
            }
            for model in sorted(allowed)
            if policy.allows(model)
        ],
    }


def _last_user_message(request: OpenAIChatRequest) -> str:
    """Extract the question from an OpenAI message list.

    Only the last user message is used as the question; earlier turns are
    conversation history the graph reconstructs from its own store. Treating the
    whole array as the question would embed the entire history into one
    retrieval query and wreck it.

    Raises:
        ValidationFailedError: when there is no user message to answer.
    """
    for message in reversed(request.messages):
        if message.role == "user" and (message.content or "").strip():
            return str(message.content)

    msg = "the messages array must contain at least one user message with content"
    raise ValidationFailedError(msg)


def _finish_reason(stop_reason: str | None) -> str:
    """Map an internal stop reason onto OpenAI's vocabulary.

    Example:
        >>> _finish_reason("budget_exhausted")
        'length'
        >>> _finish_reason("guardrail_blocked")
        'content_filter'
        >>> _finish_reason(None)
        'stop'
    """
    return {
        "budget_exhausted": "length",
        "guardrail_blocked": "content_filter",
        "error": "stop",
    }.get(stop_reason or "", "stop")
