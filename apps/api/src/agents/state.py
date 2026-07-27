"""Agent state and the per-turn budget.

The state is a single typed object threaded through every node. Two properties
make the graph tractable:

* **Nodes never mutate it.** Each returns a partial update that LangGraph merges.
  That is what makes a turn replayable from its node trace, and what stops a
  node deep in the graph quietly changing something an earlier node depends on.
* **The budget is part of the state, not ambient.** Every node that spends
  tokens or a tool call debits it, and the routing functions read it. A budget
  that lives outside the state is a budget the graph can forget to check, and an
  agent that forgets is an agent that loops.

Example:
    >>> budget = TurnBudget(max_tokens=100, max_tool_calls=2, max_iterations=3)
    >>> budget.spend_tokens(60).remaining_tokens
    40
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Annotated, Any, TypedDict

from src.guardrails.base import GuardrailResult
from src.retrieval.types import CragVerdict, RetrievedChunk
from src.services.llm.types import Message, ToolCall, Usage


class Intent(StrEnum):
    """How the router classified the turn."""

    SIMPLE_QA = "simple_qa"
    MULTI_HOP = "multi_hop"
    TOOL_USING = "tool_using"
    CLARIFICATION_NEEDED = "clarification_needed"
    OUT_OF_SCOPE = "out_of_scope"

    @classmethod
    def parse(cls, raw: str) -> Intent:
        """Parse a classifier response, defaulting to the safest label.

        Defaults to ``MULTI_HOP`` rather than ``SIMPLE_QA`` because an
        unparseable classification should cost an extra retrieval, not produce a
        confidently incomplete answer.

        Example:
            >>> Intent.parse("tool_using").value
            'tool_using'
            >>> Intent.parse("Category: SIMPLE_QA").value
            'simple_qa'
            >>> Intent.parse("something unexpected").value
            'multi_hop'
        """
        normalised = raw.strip().lower()
        for intent in cls:
            if intent.value in normalised:
                return intent
        return cls.MULTI_HOP


class StopReason(StrEnum):
    """Why the graph stopped producing output."""

    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    NO_CONTEXT = "no_context"
    CLARIFICATION_NEEDED = "clarification_needed"
    OUT_OF_SCOPE = "out_of_scope"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TurnBudget:
    """What one turn is allowed to spend.

    Immutable, so a node cannot alter the budget as a side effect: spending
    returns a new budget that must be merged into the state, which makes every
    debit visible in the state diff.
    """

    max_tokens: int = 16_000
    max_tool_calls: int = 8
    max_iterations: int = 12
    max_cost_usd: float = 1.0

    tokens_used: int = 0
    tool_calls_used: int = 0
    iterations_used: int = 0
    cost_usd: float = 0.0

    @property
    def remaining_tokens(self) -> int:
        """Tokens left, never negative."""
        return max(self.max_tokens - self.tokens_used, 0)

    @property
    def remaining_tool_calls(self) -> int:
        """Tool calls left, never negative."""
        return max(self.max_tool_calls - self.tool_calls_used, 0)

    @property
    def exhausted(self) -> bool:
        """Whether any ceiling has been reached.

        Example:
            >>> TurnBudget(max_tokens=10, tokens_used=10).exhausted
            True
            >>> TurnBudget(max_tokens=10, tokens_used=5).exhausted
            False
        """
        return (
            self.tokens_used >= self.max_tokens
            or self.tool_calls_used >= self.max_tool_calls
            or self.iterations_used >= self.max_iterations
            or self.cost_usd >= self.max_cost_usd
        )

    def spend_tokens(self, tokens: int, *, cost_usd: float = 0.0) -> TurnBudget:
        """Return a budget with tokens and cost debited."""
        return replace(
            self, tokens_used=self.tokens_used + tokens, cost_usd=self.cost_usd + cost_usd
        )

    def spend_tool_call(self, count: int = 1) -> TurnBudget:
        """Return a budget with tool calls debited."""
        return replace(self, tool_calls_used=self.tool_calls_used + count)

    def next_iteration(self) -> TurnBudget:
        """Return a budget with the iteration counter advanced."""
        return replace(self, iterations_used=self.iterations_used + 1)

    def spend_usage(self, usage: Usage, *, cost_usd: float) -> TurnBudget:
        """Debit a provider call's usage."""
        return self.spend_tokens(usage.total_tokens, cost_usd=cost_usd)


@dataclass(slots=True)
class NodeEvent:
    """One entry in the turn's execution trace.

    Surfaced to the frontend as the "thinking" panel and persisted on the message
    so a support conversation can reconstruct what the agent actually did.
    """

    node: str
    status: str
    detail: str | None = None
    duration_ms: int = 0
    data: dict[str, Any] = field(default_factory=dict)


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer that merges two dicts, right winning on conflicts."""
    return {**left, **right}


def _take_last(left: Any, right: Any) -> Any:
    """Reducer that keeps the most recent value."""
    return right if right is not None else left


class AgentState(TypedDict, total=False):
    """State threaded through the graph.

    Fields annotated with a reducer accumulate across nodes; the rest are
    last-write-wins. Accumulating the trace and the events is what lets nodes
    run concurrently without clobbering each other's telemetry.
    """

    # ── Request ──────────────────────────────────────────────────────────────
    query: str
    tenant_id: str
    user_id: str | None
    conversation_id: str | None
    message_id: str | None
    history: list[Message]
    conversation_summary: str | None
    entity_memory: dict[str, Any]

    # ── Configuration resolved for this turn ─────────────────────────────────
    model: str
    prompt_version: str | None
    tenant_instructions: str | None
    response_template: str | None

    # ── Router ───────────────────────────────────────────────────────────────
    intent: Intent | None
    intent_confidence: float | None

    # ── Rewriting ────────────────────────────────────────────────────────────
    rewritten_queries: list[str]
    sub_questions: list[str]
    hypothetical_document: str | None

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieved: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    crag_verdict: CragVerdict | None
    retrieval_expanded: bool
    web_fallback_used: bool
    top_retrieval_score: float | None

    # ── Tools ────────────────────────────────────────────────────────────────
    plan: str | None
    pending_tool_calls: list[ToolCall]
    tool_results: Annotated[list[dict[str, Any]], operator.add]

    # ── Generation ───────────────────────────────────────────────────────────
    draft_answer: str | None
    answer: str | None
    citations: list[dict[str, Any]]
    critique: str | None
    critique_verdict: str | None
    revision_count: int

    # ── Guardrails ───────────────────────────────────────────────────────────
    input_guardrails: Annotated[list[GuardrailResult], operator.add]
    output_guardrails: Annotated[list[GuardrailResult], operator.add]
    blocked_reason: str | None

    # ── Bookkeeping ──────────────────────────────────────────────────────────
    budget: TurnBudget
    node_trace: Annotated[list[str], operator.add]
    events: Annotated[list[NodeEvent], operator.add]
    stop_reason: StopReason | None
    error: str | None
    cache_hit: str | None


def initial_state(
    *,
    query: str,
    tenant_id: str,
    model: str,
    budget: TurnBudget | None = None,
    **extra: Any,
) -> AgentState:
    """Build a state for a new turn with every accumulator initialised.

    Initialising the list fields matters: LangGraph's ``operator.add`` reducers
    need a list to add to, and a missing key surfaces as a confusing type error
    several nodes later.

    Example:
        >>> state = initial_state(query="hi", tenant_id="t", model="m")
        >>> state["node_trace"], state["revision_count"]
        ([], 0)
    """
    state: AgentState = {
        "query": query,
        "tenant_id": tenant_id,
        "model": model,
        "user_id": None,
        "conversation_id": None,
        "message_id": None,
        "history": [],
        "conversation_summary": None,
        "entity_memory": {},
        "prompt_version": None,
        "tenant_instructions": None,
        "response_template": None,
        "intent": None,
        "intent_confidence": None,
        "rewritten_queries": [],
        "sub_questions": [],
        "hypothetical_document": None,
        "retrieved": [],
        "reranked": [],
        "crag_verdict": None,
        "retrieval_expanded": False,
        "web_fallback_used": False,
        "top_retrieval_score": None,
        "plan": None,
        "pending_tool_calls": [],
        "tool_results": [],
        "draft_answer": None,
        "answer": None,
        "citations": [],
        "critique": None,
        "critique_verdict": None,
        "revision_count": 0,
        "input_guardrails": [],
        "output_guardrails": [],
        "blocked_reason": None,
        "budget": budget or TurnBudget(),
        "node_trace": [],
        "events": [],
        "stop_reason": None,
        "error": None,
        "cache_hit": None,
    }
    state.update(extra)  # type: ignore[typeddict-item]
    return state


def context_chunks(state: AgentState) -> list[RetrievedChunk]:
    """The chunks the generator should actually cite.

    Reranked results win when present, because reranking is the last word on
    ordering; otherwise the fused retrieval stands.

    Example:
        >>> context_chunks({"retrieved": [], "reranked": []})
        []
    """
    return list(state.get("reranked") or state.get("retrieved") or [])
