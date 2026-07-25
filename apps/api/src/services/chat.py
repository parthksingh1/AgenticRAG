"""Chat service: runs a turn and persists it.

The service layer is where the graph meets the database. Routes hold no logic
and repositories hold no policy, which leaves exactly one place that decides
what a turn *means*: what gets saved, what gets measured, and what the user is
told when something goes wrong.

The ordering inside :meth:`ChatService.stream` is the load-bearing part. The user
message is committed before the graph runs, and the assistant message is
committed as soon as the graph finishes — before the stream is closed. A client
that disconnects mid-answer therefore loses nothing, which is the difference
between a flaky connection costing a repaint and costing a full turn.

Example:
    >>> from src.services.chat import ChatService
    >>> ChatService.__name__
    'ChatService'
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from src.agents.state import StopReason, TurnBudget
from src.core.errors import NotFoundError
from src.core.logging import get_logger
from src.models.conversation import MessageRole, MessageStatus
from src.schemas.chat import ChatRequest, ChatResponse, CitationOut, StreamEventOut
from src.services.llm.types import Message as LLMMessage

if TYPE_CHECKING:
    from src.api.auth import Principal
    from src.api.dependencies import TenantRuntime

log = get_logger(__name__)

#: Turns kept verbatim in the prompt. Older ones are represented by the running
#: summary, which is what stops a long conversation's cost growing without bound.
HISTORY_WINDOW = 10

#: Summarisation runs once a conversation is this many turns past its last
#: summary. Running it every turn would double the cost of a long conversation
#: for a summary that barely changes.
SUMMARISE_EVERY = 6


class ChatService:
    """Runs one chat turn end to end."""

    def __init__(self, *, runtime: TenantRuntime, principal: Principal) -> None:
        """Bind the service to a tenant runtime and a caller."""
        self._runtime = runtime
        self._principal = principal

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Run a turn and return the finished answer."""
        started = time.perf_counter()
        conversation, user_message, history = await self._begin(request)

        state = await self._runtime.agent.run(
            request.message,
            tenant_id=self._runtime.tenant_id,
            model=self._resolve_model(request),
            budget=self._budget(),
            history=history,
            conversation_id=conversation.id,
            user_id=self._principal.user_id,
            tenant_instructions=self._runtime.tenant.custom_instructions,
            response_template=self._runtime.tenant.response_template,
            conversation_summary=conversation.running_summary,
            prompt_version=request.prompt_version,
        )

        message = await self._finish(
            conversation=conversation,
            parent_id=user_message.id,
            state=state,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return self._to_response(
            message_id=message["id"], conversation_id=conversation.id, state=state, timings=message
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEventOut]:
        """Run a turn, yielding progress and tokens as they happen."""
        started = time.perf_counter()
        conversation, user_message, history = await self._begin(request)

        yield StreamEventOut(
            type="start", conversation_id=conversation.id, message_id=user_message.id
        )

        first_token_at: float | None = None
        state: dict[str, Any] = {}

        async for event in self._runtime.agent.stream(
            request.message,
            tenant_id=self._runtime.tenant_id,
            model=self._resolve_model(request),
            budget=self._budget(),
            history=history,
            conversation_id=conversation.id,
            user_id=self._principal.user_id,
            tenant_instructions=self._runtime.tenant.custom_instructions,
            response_template=self._runtime.tenant.response_template,
            conversation_summary=conversation.running_summary,
            prompt_version=request.prompt_version,
        ):
            kind = event.get("type")

            if kind == "token":
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                yield StreamEventOut(type="token", text=event.get("text", ""))

            elif kind in ("node_start", "node_end") and request.include_thinking:
                yield StreamEventOut(
                    type="node",
                    node=event.get("node"),
                    status="running" if kind == "node_start" else "done",
                    data=event.get("output"),
                )

            elif kind == "state":
                state = event.get("state", {})

        # The graph's streaming interface reports progress; the final state is
        # fetched once here so persistence sees the complete turn rather than a
        # reconstruction from events.
        if not state:
            state = await self._runtime.agent.run(
                request.message,
                tenant_id=self._runtime.tenant_id,
                model=self._resolve_model(request),
                budget=self._budget(),
                history=history,
                conversation_id=conversation.id,
                user_id=self._principal.user_id,
                tenant_instructions=self._runtime.tenant.custom_instructions,
                response_template=self._runtime.tenant.response_template,
                conversation_summary=conversation.running_summary,
                prompt_version=request.prompt_version,
            )
            answer = state.get("answer") or ""
            if answer:
                yield StreamEventOut(type="token", text=answer)
                first_token_at = first_token_at or time.perf_counter()

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        ttft_ms = int((first_token_at - started) * 1000) if first_token_at else None

        # Persisted before the stream closes: a client that disconnects here has
        # already had its answer saved.
        message = await self._finish(
            conversation=conversation,
            parent_id=user_message.id,
            state=state,
            elapsed_ms=elapsed_ms,
            ttft_ms=ttft_ms,
        )

        citations = self._citations(state)
        if citations:
            yield StreamEventOut(type="citations", citations=citations)

        yield StreamEventOut(
            type="done",
            message_id=message["id"],
            conversation_id=conversation.id,
            citations=citations,
            usage={
                "prompt_tokens": message["prompt_tokens"],
                "completion_tokens": message["completion_tokens"],
                "cost_usd": message["cost_usd"],
                "ttft_ms": ttft_ms,
                "ttlt_ms": elapsed_ms,
                "cache_hit": state.get("cache_hit"),
            },
        )

    async def regenerate(self, *, message_id: str, model: str | None = None) -> ChatResponse:
        """Regenerate an answer as a sibling of the original.

        Raises:
            NotFoundError: when the message does not exist in this workspace.
        """
        from src.core.db import session_scope
        from src.repositories.conversations import load_message

        async with session_scope() as session:
            original = await load_message(
                session, message_id=message_id, tenant_id=self._runtime.tenant_id
            )
        if original is None or not original.parent_message_id:
            raise NotFoundError("That message cannot be regenerated.")

        async with session_scope() as session:
            parent = await load_message(
                session,
                message_id=original.parent_message_id,
                tenant_id=self._runtime.tenant_id,
            )
        if parent is None:
            raise NotFoundError("The original question could not be found.")

        return await self.complete(
            ChatRequest(
                message=parent.content,
                conversation_id=None,
                parent_message_id=parent.id,
                model=model,
                stream=False,
            )
        )

    # ── internals ────────────────────────────────────────────────────────────

    async def _begin(self, request: ChatRequest) -> tuple[Any, Any, list[LLMMessage]]:
        """Load or create the conversation and persist the user's message."""
        from src.core.db import session_scope
        from src.repositories.conversations import (
            append_message,
            get_or_create_conversation,
            history_for,
        )

        async with session_scope() as session:
            conversation = await get_or_create_conversation(
                session,
                conversation_id=request.conversation_id,
                tenant_id=self._runtime.tenant_id,
                user_id=self._principal.user_id,
                first_message=request.message,
                model=self._resolve_model(request),
            )
            history_rows = await history_for(
                session, conversation=conversation, window=HISTORY_WINDOW
            )
            user_message = await append_message(
                session,
                conversation=conversation,
                role=MessageRole.USER,
                content=request.message,
                parent_message_id=request.parent_message_id,
            )
            # Detached copies: the session closes on exit, and touching a lazy
            # attribute afterwards would raise deep inside the graph.
            snapshot = _detach(conversation)
            user_snapshot = _detach(user_message)
            history = [
                LLMMessage(role=_role_for(m.role), content=m.content)
                for m in history_rows
                if m.content
            ]

        return snapshot, user_snapshot, history

    async def _finish(
        self,
        *,
        conversation: Any,
        parent_id: str,
        state: dict[str, Any],
        elapsed_ms: int,
        ttft_ms: int | None = None,
    ) -> dict[str, Any]:
        """Persist the assistant message, its citations and its telemetry."""
        from src.core.db import session_scope
        from src.observability.metrics import record_answer, record_cache
        from src.observability.tracing import current_trace_id, langfuse_trace_url
        from src.repositories.conversations import (
            append_message,
            get_or_create_conversation,
            save_citations,
        )

        budget: TurnBudget = state.get("budget") or TurnBudget()
        stop_reason = state.get("stop_reason")
        citations = state.get("citations") or []
        trace_id = current_trace_id()

        async with session_scope() as session:
            live = await get_or_create_conversation(
                session,
                conversation_id=conversation.id,
                tenant_id=self._runtime.tenant_id,
                user_id=self._principal.user_id,
                first_message="",
            )
            message = await append_message(
                session,
                conversation=live,
                role=MessageRole.ASSISTANT,
                content=state.get("answer") or "",
                parent_message_id=parent_id,
                status=(
                    MessageStatus.BLOCKED
                    if stop_reason is StopReason.GUARDRAIL_BLOCKED
                    else MessageStatus.COMPLETE
                ),
                model=state.get("model"),
                intent=getattr(state.get("intent"), "value", None),
                prompt_version=state.get("prompt_version"),
                node_trace=list(state.get("node_trace") or []),
                strategies_used=list(state.get("strategies_used") or []),
                prompt_tokens=budget.tokens_used,
                completion_tokens=0,
                cost_usd=budget.cost_usd,
                ttft_ms=ttft_ms,
                ttlt_ms=elapsed_ms,
                cache_hit=state.get("cache_hit"),
                groundedness_score=_groundedness(state),
                guardrail_flags=_flags(state),
                trace_id=trace_id,
                langfuse_trace_url=langfuse_trace_url(trace_id, self._runtime.services.settings),
                error_message=state.get("error"),
            )
            await save_citations(session, message=message, citations=citations)

            live.total_tokens += budget.tokens_used
            live.total_cost_usd = float(live.total_cost_usd or 0) + budget.cost_usd

            persisted = {
                "id": message.id,
                "prompt_tokens": budget.tokens_used,
                "completion_tokens": 0,
                "cost_usd": budget.cost_usd,
                "ttft_ms": ttft_ms,
                "ttlt_ms": elapsed_ms,
            }

        record_answer(
            model=state.get("model") or self._runtime.model,
            ttft_ms=ttft_ms,
            ttlt_ms=elapsed_ms,
            completion_tokens=0,
            groundedness=_groundedness(state),
        )
        if state.get("cache_hit"):
            record_cache(cache="semantic", hit=True)

        await self._maybe_cache_answer(state)
        return persisted

    async def _maybe_cache_answer(self, state: dict[str, Any]) -> None:
        """Cache a clean, well-grounded answer for reuse.

        Only answers that were neither blocked nor served from cache, and that
        carry at least one verified citation. Caching an ungrounded answer would
        multiply one bad answer across every similar future question.
        """
        cache = self._runtime.services.semantic_cache
        if (
            cache is None
            or state.get("cache_hit")
            or state.get("stop_reason")
            not in (
                StopReason.COMPLETED,
                None,
            )
        ):
            return

        citations = state.get("citations") or []
        if not any(c.get("verified") for c in citations):
            return

        try:
            await cache.store(
                state.get("query", ""),
                {"answer": state.get("answer"), "citations": citations},
                tenant_id=self._runtime.tenant_id,
                model=state.get("model"),
                prompt_version=state.get("prompt_version"),
            )
        except Exception as exc:  # noqa: BLE001 - caching must not fail a turn
            log.warning("failed to cache answer", reason=str(exc))

    def _resolve_model(self, request: ChatRequest) -> str:
        """Choose the model for this turn, honouring the workspace policy."""
        return request.model or self._runtime.model

    def _budget(self) -> TurnBudget:
        """Build the per-turn budget from settings."""
        settings = self._runtime.services.settings
        return TurnBudget(
            max_tokens=settings.max_tokens_per_request,
            max_tool_calls=settings.max_tool_calls_per_turn,
            max_iterations=settings.max_agent_iterations,
        )

    @staticmethod
    def _citations(state: dict[str, Any]) -> tuple[CitationOut, ...]:
        """Map the graph's citations onto their API shape."""
        return tuple(
            CitationOut(
                marker=int(c.get("marker", 0)),
                chunk_id=c.get("chunk_id"),
                document_id=c.get("document_id"),
                document_title=c.get("document_title"),
                page_number=c.get("page_number"),
                quote=c.get("claim"),
                verified=bool(c.get("verified")),
                score=c.get("score"),
            )
            for c in (state.get("citations") or [])
        )

    def _to_response(
        self,
        *,
        message_id: str,
        conversation_id: str,
        state: dict[str, Any],
        timings: dict[str, Any],
    ) -> ChatResponse:
        """Assemble the non-streaming response."""
        from src.observability.tracing import current_trace_id, langfuse_trace_url

        trace_id = current_trace_id()
        return ChatResponse(
            message_id=message_id,
            conversation_id=conversation_id,
            content=state.get("answer") or "",
            citations=self._citations(state),
            model=state.get("model") or self._runtime.model,
            intent=getattr(state.get("intent"), "value", None),
            node_trace=tuple(state.get("node_trace") or ()),
            strategies_used=tuple(state.get("strategies_used") or ()),
            prompt_tokens=timings["prompt_tokens"],
            completion_tokens=timings["completion_tokens"],
            cost_usd=timings["cost_usd"],
            ttft_ms=timings.get("ttft_ms"),
            ttlt_ms=timings.get("ttlt_ms"),
            cache_hit=state.get("cache_hit"),
            groundedness=_groundedness(state),
            guardrail_flags=tuple(_flags(state)),
            stop_reason=getattr(state.get("stop_reason"), "value", None),
            trace_id=trace_id,
            langfuse_url=langfuse_trace_url(trace_id, self._runtime.services.settings),
        )


def _role_for(role: MessageRole) -> Any:
    """Map a stored role onto the provider-facing role."""
    from src.services.llm.types import Role

    return {
        MessageRole.USER: Role.USER,
        MessageRole.ASSISTANT: Role.ASSISTANT,
        MessageRole.SYSTEM: Role.SYSTEM,
        MessageRole.TOOL: Role.TOOL,
    }[role]


def _groundedness(state: dict[str, Any]) -> float | None:
    """Extract the groundedness score from the output guardrail results.

    Example:
        >>> _groundedness({}) is None
        True
    """
    for result in state.get("output_guardrails") or []:
        if getattr(result, "kind", None) is not None and result.score is not None:
            evidence = getattr(result, "evidence", {}) or {}
            if "citation_recall" in evidence:
                return float(evidence["citation_recall"])
    return None


def _flags(state: dict[str, Any]) -> list[str]:
    """Every guardrail kind that fired on this turn.

    Example:
        >>> _flags({})
        []
    """
    from src.models.telemetry import GuardrailDecision

    kinds = {
        result.kind.value
        for stage in ("input_guardrails", "output_guardrails")
        for result in (state.get(stage) or [])
        if result.decision is not GuardrailDecision.ALLOW
    }
    return sorted(kinds)


def _detach(instance: Any) -> Any:
    """Snapshot an ORM row's scalar fields into a plain object.

    Passing a live ORM instance out of its session and touching a lazy attribute
    later raises ``DetachedInstanceError`` somewhere far from the cause; taking a
    snapshot makes that impossible.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        **{c.name: getattr(instance, c.name) for c in instance.__table__.columns}
    )
