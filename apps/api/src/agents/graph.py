"""The LangGraph state machine.

The shape of the graph is the product decision. Everything routes through
guardrails and the cache first — before any money is spent — and every path
converges on the same output guardrails, so there is no route by which an answer
reaches the user unscreened.

::

    input_guardrails
        ├─ blocked ──────────────────────────────────────────────▶ formatter
        └─ ok ──▶ cache_lookup
                    ├─ hit ───────────────────────────────────────▶ formatter
                    └─ miss ──▶ intent_router
                                  ├─ out_of_scope / clarification ─▶ formatter
                                  ├─ tool_using ──▶ planner ──▶ tool_executor ─┐
                                  └─ qa / multi_hop ──▶ query_rewriter          │
                                                          ──▶ retriever         │
                                                          ──▶ reranker ─────────┤
                                                                                ▼
                                                                          generator
                                                                                │
                                                                      citation_binder
                                                                                │
                                                                        self_critic
                                                                     ├─ REVISE ─┘ (max 2)
                                                                     └─ accept
                                                                                ▼
                                                                     output_guardrails
                                                                                ▼
                                                                           formatter

Three routing decisions are worth stating explicitly:

* **Blocked turns still pass through the formatter.** A refusal is a response,
  and giving it the same tenant formatting as any other keeps the frontend from
  needing a second rendering path.
* **The critique loop is bounded twice** — by ``MAX_REVISIONS`` and by the turn
  budget. Either alone is insufficient: a budget-only bound lets a cheap model
  loop a dozen times, and a count-only bound lets an expensive one blow the cost
  ceiling.
* **Tool turns rejoin the same generator.** Tool output becomes context
  alongside retrieved chunks rather than taking a separate answer path, so
  citation verification applies to it too.

Example:
    >>> from src.agents.graph import route_after_intent
    >>> from src.agents.state import Intent
    >>> route_after_intent({"intent": Intent.TOOL_USING, "budget": None})
    'planner'
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, StateGraph

from src.agents.nodes import (
    MAX_REVISIONS,
    NodeDependencies,
    make_cache_lookup_node,
    make_citation_binder_node,
    make_formatter_node,
    make_generator_node,
    make_input_guardrails_node,
    make_intent_router_node,
    make_output_guardrails_node,
    make_planner_node,
    make_query_rewriter_node,
    make_reranker_node,
    make_retriever_node,
    make_self_critic_node,
    make_tool_executor_node,
)
from src.agents.state import AgentState, Intent, TurnBudget, initial_state
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.services.llm.types import Message

log = get_logger(__name__)


# ── Routing functions ────────────────────────────────────────────────────────


def route_after_input_guardrails(state: AgentState) -> str:
    """Skip everything when the input was blocked.

    Example:
        >>> from src.agents.state import StopReason
        >>> route_after_input_guardrails({"stop_reason": StopReason.GUARDRAIL_BLOCKED})
        'formatter'
        >>> route_after_input_guardrails({})
        'cache_lookup'
    """
    return "formatter" if state.get("stop_reason") else "cache_lookup"


def route_after_cache(state: AgentState) -> str:
    """A cache hit is already an answer; go straight to formatting.

    Example:
        >>> route_after_cache({"cache_hit": "semantic"})
        'formatter'
        >>> route_after_cache({})
        'intent_router'
    """
    return "formatter" if state.get("cache_hit") else "intent_router"


def route_after_intent(state: AgentState) -> str:
    """Send the turn down the path its intent calls for.

    Out-of-scope and clarification turns produce a response without retrieval,
    because retrieving for a question the corpus cannot answer wastes a round
    trip to reach the same refusal.

    Example:
        >>> route_after_intent({"intent": Intent.SIMPLE_QA})
        'query_rewriter'
        >>> route_after_intent({"intent": Intent.OUT_OF_SCOPE})
        'formatter'
    """
    intent = state.get("intent")
    if intent is Intent.OUT_OF_SCOPE:
        return "formatter"
    if intent is Intent.CLARIFICATION_NEEDED:
        return "formatter"
    if intent is Intent.TOOL_USING:
        return "planner"
    return "query_rewriter"


def route_after_critique(state: AgentState) -> str:
    """Loop back to the generator only when a revision is both wanted and affordable.

    Example:
        >>> budget = TurnBudget(max_tokens=1000, tokens_used=10)
        >>> route_after_critique({"critique_verdict": "REVISE", "revision_count": 1,
        ...                       "budget": budget})
        'generator'
        >>> route_after_critique({"critique_verdict": "ACCEPT", "revision_count": 0,
        ...                       "budget": budget})
        'output_guardrails'
    """
    if state.get("critique_verdict") != "REVISE":
        return "output_guardrails"
    if state.get("revision_count", 0) > MAX_REVISIONS:
        return "output_guardrails"
    budget = state.get("budget")
    if budget is not None and budget.exhausted:
        return "output_guardrails"
    return "generator"


# ── Graph construction ───────────────────────────────────────────────────────


def build_graph(deps: NodeDependencies) -> Any:
    """Compile the agent graph for a set of dependencies.

    Returns a compiled LangGraph app exposing ``ainvoke`` and ``astream_events``.
    """
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("input_guardrails", make_input_guardrails_node(deps))
    graph.add_node("cache_lookup", make_cache_lookup_node(deps))
    graph.add_node("intent_router", make_intent_router_node(deps))
    graph.add_node("query_rewriter", make_query_rewriter_node(deps))
    graph.add_node("retriever", make_retriever_node(deps))
    graph.add_node("reranker", make_reranker_node(deps))
    graph.add_node("planner", make_planner_node(deps))
    graph.add_node("tool_executor", make_tool_executor_node(deps))
    graph.add_node("generator", make_generator_node(deps))
    graph.add_node("citation_binder", make_citation_binder_node(deps))
    graph.add_node("self_critic", make_self_critic_node(deps))
    graph.add_node("output_guardrails", make_output_guardrails_node(deps))
    graph.add_node("formatter", make_formatter_node(deps))

    graph.set_entry_point("input_guardrails")

    graph.add_conditional_edges(
        "input_guardrails",
        route_after_input_guardrails,
        {"formatter": "formatter", "cache_lookup": "cache_lookup"},
    )
    graph.add_conditional_edges(
        "cache_lookup",
        route_after_cache,
        {"formatter": "formatter", "intent_router": "intent_router"},
    )
    graph.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "formatter": "formatter",
            "planner": "planner",
            "query_rewriter": "query_rewriter",
        },
    )

    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "generator")
    graph.add_edge("planner", "tool_executor")
    # Tool turns still retrieve: a question that needs a calculation usually also
    # needs the document that says which numbers to calculate with.
    graph.add_edge("tool_executor", "retriever")

    graph.add_edge("generator", "citation_binder")
    graph.add_edge("citation_binder", "self_critic")
    graph.add_conditional_edges(
        "self_critic",
        route_after_critique,
        {"generator": "generator", "output_guardrails": "output_guardrails"},
    )
    graph.add_edge("output_guardrails", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()


class AgentRunner:
    """Runs the compiled graph for one turn."""

    def __init__(self, deps: NodeDependencies) -> None:
        """Compile the graph once and reuse it across turns."""
        self._deps = deps
        self._app = build_graph(deps)

    async def run(
        self,
        query: str,
        *,
        tenant_id: str,
        model: str,
        budget: TurnBudget | None = None,
        history: list[Message] | None = None,
        **extra: Any,
    ) -> AgentState:
        """Execute one turn and return the final state."""
        state = initial_state(
            query=query,
            tenant_id=tenant_id,
            model=model,
            budget=budget,
            history=history or [],
            **extra,
        )
        # The recursion limit is a backstop behind the budget and MAX_REVISIONS;
        # reaching it means a routing bug, not an expensive question.
        result = await self._app.ainvoke(state, config={"recursion_limit": 40})
        return dict(result)  # type: ignore[return-value]

    async def stream(
        self,
        query: str,
        *,
        tenant_id: str,
        model: str,
        budget: TurnBudget | None = None,
        history: list[Message] | None = None,
        **extra: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream node-level progress events for one turn.

        Yields a dict per event so the SSE layer can serialise it directly. The
        frontend renders these as the thinking panel, which is why every node
        emits one even when it does nothing.
        """
        state = initial_state(
            query=query,
            tenant_id=tenant_id,
            model=model,
            budget=budget,
            history=history or [],
            **extra,
        )
        async for event in self._app.astream_events(
            state, version="v2", config={"recursion_limit": 40}
        ):
            kind = event.get("event")
            if kind == "on_chain_start" and event.get("name") in _STREAMED_NODES:
                yield {"type": "node_start", "node": event["name"]}
            elif kind == "on_chain_end" and event.get("name") in _STREAMED_NODES:
                yield {
                    "type": "node_end",
                    "node": event["name"],
                    "output": _summarise(event.get("data", {}).get("output")),
                }
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                text = getattr(chunk, "content", None)
                if text:
                    yield {"type": "token", "text": text}


_STREAMED_NODES = frozenset(
    {
        "input_guardrails",
        "cache_lookup",
        "intent_router",
        "query_rewriter",
        "retriever",
        "reranker",
        "planner",
        "tool_executor",
        "generator",
        "citation_binder",
        "self_critic",
        "output_guardrails",
        "formatter",
    }
)


def _summarise(output: Any) -> dict[str, Any]:
    """Reduce a node's output to what the thinking panel needs.

    Sending the full state on every node would push megabytes of chunk text down
    the SSE stream for a single turn.

    Example:
        >>> _summarise({"retrieved": [1, 2, 3], "answer": "x" * 500})["retrieved"]
        3
    """
    if not isinstance(output, dict):
        return {}

    summary: dict[str, Any] = {}
    for key in ("intent", "crag_verdict", "cache_hit", "critique_verdict", "stop_reason"):
        value = output.get(key)
        if value is not None:
            summary[key] = getattr(value, "value", value)
    for key in ("retrieved", "reranked", "citations", "tool_results", "rewritten_queries"):
        value = output.get(key)
        if value is not None:
            summary[key] = len(value)
    if output.get("events"):
        summary["detail"] = output["events"][-1].detail
    return summary
