"""Graph nodes.

Each node is an async callable taking the state and returning a *partial* update.
None of them mutate the state they are given, which is what makes a turn
replayable from its trace and stops a late node quietly changing something an
early node established.

Two conventions run through all of them:

* **A node failure degrades the turn, it does not end it.** A node that raises
  would abort a turn the user is watching; instead each catches, records the
  error in its event, and returns whatever partial progress it has. The graph's
  routing then decides whether the turn can still produce an answer.
* **Every node debits the budget.** Nodes that call a model debit tokens and
  cost; the tool executor debits tool calls; the loop entry points debit an
  iteration. That is what makes the agent terminate.

Example:
    >>> from src.agents.nodes import NODE_NAMES
    >>> "generator" in NODE_NAMES
    True
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.agents.state import (
    AgentState,
    Intent,
    NodeEvent,
    StopReason,
    TurnBudget,
    context_chunks,
)
from src.core.logging import get_logger
from src.guardrails.base import GuardrailContext, GuardrailPipeline, GuardrailPolicy
from src.guardrails.injection import neutralise_chunk, scan_retrieved_context
from src.retrieval.types import RetrievalRequest, RetrievedChunk
from src.services.llm.types import CompletionRequest, Message, ToolCall

if TYPE_CHECKING:
    from src.caching.semantic import SemanticCache
    from src.guardrails.groundedness import CitationVerifier
    from src.mcp_clients.registry import ToolRegistry
    from src.retrieval.hybrid import HybridRetriever
    from src.services.llm.router import LLMRouter
    from src.services.prompts import PromptRegistry

log = get_logger(__name__)

NODE_NAMES = (
    "input_guardrails",
    "cache_lookup",
    "intent_router",
    "query_rewriter",
    "retriever",
    "retrieval_evaluator",
    "reranker",
    "planner",
    "tool_executor",
    "generator",
    "citation_binder",
    "self_critic",
    "output_guardrails",
    "formatter",
)

#: A critique loop that never converges is worse than a slightly flawed answer.
MAX_REVISIONS = 2


@dataclass(slots=True)
class NodeDependencies:
    """Everything the nodes need, injected once when the graph is built.

    Passing these through the constructor rather than importing them keeps the
    nodes pure enough to test individually, and lets the eval harness swap in
    fakes for the whole stack.
    """

    router: LLMRouter
    prompts: PromptRegistry
    retriever: HybridRetriever | None = None
    tools: ToolRegistry | None = None
    input_guardrails: GuardrailPipeline | None = None
    output_guardrails: GuardrailPipeline | None = None
    citation_verifier: CitationVerifier | None = None
    semantic_cache: SemanticCache | None = None
    policy: GuardrailPolicy | None = None
    top_k: int = 5

    def guardrail_context(self, state: AgentState, **extra: Any) -> GuardrailContext:
        """Build the guardrail context for a turn."""
        return GuardrailContext(
            tenant_id=state["tenant_id"],
            policy=self.policy or GuardrailPolicy(),
            user_id=state.get("user_id"),
            conversation_id=state.get("conversation_id"),
            message_id=state.get("message_id"),
            query=state.get("query"),
            **extra,
        )


def _event(node: str, status: str, started: float, **data: Any) -> NodeEvent:
    """Build a trace event with its measured duration."""
    detail = data.pop("detail", None)
    return NodeEvent(
        node=node,
        status=status,
        detail=detail,
        duration_ms=int((time.perf_counter() - started) * 1000),
        data=data,
    )


async def _complete(
    deps: NodeDependencies,
    state: AgentState,
    *,
    messages: Sequence[Message],
    node: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> tuple[str, TurnBudget]:
    """Run one model call and return its text plus the debited budget."""
    request = CompletionRequest(
        messages=tuple(messages),
        model=state["model"],
        max_tokens=max_tokens,
        temperature=temperature,
        tenant_id=state["tenant_id"],
        node=node,
        prompt_version=state.get("prompt_version"),
    )
    completion = await deps.router.complete(request)
    budget = state["budget"].spend_usage(completion.usage, cost_usd=completion.cost_usd)
    return completion.content, budget


# ── Nodes ────────────────────────────────────────────────────────────────────


def make_input_guardrails_node(deps: NodeDependencies) -> Callable[[AgentState], Awaitable[dict]]:
    """Screen the user's message before anything else spends money."""

    async def input_guardrails(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        if deps.input_guardrails is None:
            return {"node_trace": ["input_guardrails"]}

        outcome = await deps.input_guardrails.run(
            state["query"], context=deps.guardrail_context(state)
        )
        update: dict[str, Any] = {
            "node_trace": ["input_guardrails"],
            "input_guardrails": list(outcome.results),
            "events": [_event("input_guardrails", "ok", started, flags=list(outcome.flags))],
        }
        if outcome.blocked:
            blocking = outcome.blocking_result
            update["blocked_reason"] = blocking.reason if blocking else "blocked by policy"
            update["stop_reason"] = StopReason.GUARDRAIL_BLOCKED
            update["answer"] = _refusal_for(blocking.kind.value if blocking else "policy")
        elif outcome.was_modified:
            # Redaction rewrites the query the rest of the graph sees, so PII
            # never reaches a provider or a log line downstream.
            update["query"] = outcome.text
        return update

    return input_guardrails


def make_cache_lookup_node(deps: NodeDependencies) -> Callable[[AgentState], Awaitable[dict]]:
    """Serve a semantically equivalent cached answer when one is safe to use."""

    async def cache_lookup(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        if deps.semantic_cache is None:
            return {"node_trace": ["cache_lookup"]}

        hit = await deps.semantic_cache.lookup(state["query"], tenant_id=state["tenant_id"])
        if hit is None:
            return {
                "node_trace": ["cache_lookup"],
                "events": [_event("cache_lookup", "miss", started)],
            }

        payload = (
            hit.entry.value if isinstance(hit.entry.value, dict) else {"answer": hit.entry.value}
        )
        return {
            "node_trace": ["cache_lookup"],
            "answer": payload.get("answer"),
            "citations": payload.get("citations", []),
            "cache_hit": "semantic",
            "stop_reason": StopReason.COMPLETED,
            "events": [_event("cache_lookup", "hit", started, similarity=round(hit.similarity, 4))],
        }

    return cache_lookup


def make_intent_router_node(deps: NodeDependencies) -> Callable[[AgentState], Awaitable[dict]]:
    """Classify the turn so the graph can route it."""

    async def intent_router(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        prompt = deps.prompts.render(
            "intent_router",
            question=state["query"],
            conversation_summary=state.get("conversation_summary"),
            has_tools=deps.tools is not None,
        )
        try:
            text, budget = await _complete(
                deps, state, messages=[Message.user(prompt)], node="intent_router", max_tokens=12
            )
        except Exception as exc:  # noqa: BLE001 - a classifier outage must not end the turn
            log.warning("intent routing failed; assuming multi_hop", reason=str(exc))
            return {
                "node_trace": ["intent_router"],
                "intent": Intent.MULTI_HOP,
                "events": [_event("intent_router", "degraded", started, detail=str(exc))],
            }

        intent = Intent.parse(text)
        return {
            "node_trace": ["intent_router"],
            "intent": intent,
            "budget": budget.next_iteration(),
            "events": [_event("intent_router", "ok", started, intent=intent.value)],
        }

    return intent_router


def make_query_rewriter_node(deps: NodeDependencies) -> Callable[[AgentState], Awaitable[dict]]:
    """Expand the query with HyDE, paraphrases or sub-questions.

    Rewriting is delegated to the retriever's own rewriter so there is one
    implementation; this node exists to record the result in the state where the
    thinking panel and the retrieval log can see it.
    """

    async def query_rewriter(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        rewriter = getattr(deps.retriever, "_rewriter", None) if deps.retriever else None
        if rewriter is None:
            return {"node_trace": ["query_rewriter"]}

        result = await rewriter.rewrite(
            state["query"],
            use_hyde=True,
            use_multi_query=True,
            decompose=state.get("intent") is Intent.MULTI_HOP,
        )
        return {
            "node_trace": ["query_rewriter"],
            "rewritten_queries": list(result.expansions),
            "sub_questions": list(result.sub_questions),
            "hypothetical_document": result.hypothetical_document,
            "budget": state["budget"].spend_tokens(0, cost_usd=result.cost_usd),
            "events": [
                _event(
                    "query_rewriter",
                    "ok",
                    started,
                    techniques=list(result.techniques),
                    variants=len(result.expansions),
                )
            ],
        }

    return query_rewriter


def make_retriever_node(deps: NodeDependencies) -> Callable[[AgentState], Awaitable[dict]]:
    """Run hybrid retrieval and scan the result for indirect injection."""

    async def retriever(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        if deps.retriever is None:
            return {"node_trace": ["retriever"], "stop_reason": StopReason.NO_CONTEXT}

        request = RetrievalRequest(
            query=state["query"],
            top_k=deps.top_k,
            expansions=tuple(state.get("rewritten_queries") or ()),
        )
        try:
            result = await deps.retriever.retrieve(request, tenant_id=state["tenant_id"])
        except Exception as exc:  # noqa: BLE001 - answer without context rather than failing
            log.error("retrieval failed", reason=str(exc))
            return {
                "node_trace": ["retriever"],
                "retrieved": [],
                "events": [_event("retriever", "failed", started, detail=str(exc))],
            }

        # An attack inside a retrieved document is the case a user-input-only
        # guardrail misses entirely, so poisoned chunks are wrapped before the
        # generator ever sees them.
        findings = scan_retrieved_context(result.chunks)
        poisoned = {f["chunk_id"] for f in findings}
        chunks = [
            c.model_copy(update={"content": neutralise_chunk(c.content)})
            if c.chunk_id in poisoned
            else c
            for c in result.chunks
        ]

        return {
            "node_trace": ["retriever"],
            "retrieved": chunks,
            "crag_verdict": result.crag_verdict,
            "retrieval_expanded": result.expanded,
            "web_fallback_used": result.web_fallback_used,
            "top_retrieval_score": result.top_score,
            "events": [
                _event(
                    "retriever",
                    "ok",
                    started,
                    strategy=result.strategy,
                    results=len(chunks),
                    neutralised=len(poisoned),
                    top_score=round(result.top_score or 0.0, 4),
                )
            ],
        }

    return retriever


def make_reranker_node(deps: NodeDependencies) -> Callable[[AgentState], Awaitable[dict]]:
    """Record the retriever's reranked output in the state.

    The hybrid retriever already reranks internally; this node makes the result
    visible as its own step in the trace, which is what the thinking panel and
    the failure explorer read.
    """

    async def reranker(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        retrieved = state.get("retrieved") or []
        return {
            "node_trace": ["reranker"],
            "reranked": list(retrieved[: deps.top_k]),
            "events": [_event("reranker", "ok", started, kept=min(len(retrieved), deps.top_k))],
        }

    return reranker


def make_planner_node(deps: NodeDependencies) -> Callable[[AgentState], Awaitable[dict]]:
    """Plan the tool sequence for a tool-using turn."""

    async def planner(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        if deps.tools is None:
            return {"node_trace": ["planner"]}

        prompt = deps.prompts.render(
            "planner",
            question=state["query"],
            tools=deps.tools.describe(),
            max_steps=state["budget"].remaining_tool_calls,
            conversation_summary=state.get("conversation_summary"),
        )
        try:
            plan, budget = await _complete(
                deps, state, messages=[Message.user(prompt)], node="planner", max_tokens=400
            )
        except Exception as exc:  # noqa: BLE001 - fall through to unplanned execution
            log.warning("planning failed; executing without a plan", reason=str(exc))
            return {
                "node_trace": ["planner"],
                "events": [_event("planner", "degraded", started, detail=str(exc))],
            }

        return {
            "node_trace": ["planner"],
            "plan": plan,
            "budget": budget.next_iteration(),
            "events": [_event("planner", "ok", started, detail=plan[:200])],
        }

    return planner


def make_tool_executor_node(deps: NodeDependencies) -> Callable[[AgentState], Awaitable[dict]]:
    """Run the model's requested tools until it stops asking or the budget ends."""

    async def tool_executor(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        if deps.tools is None:
            return {"node_trace": ["tool_executor"]}

        budget = state["budget"].next_iteration()
        if budget.exhausted:
            return {
                "node_trace": ["tool_executor"],
                "budget": budget,
                "stop_reason": StopReason.BUDGET_EXHAUSTED,
                "events": [_event("tool_executor", "budget_exhausted", started)],
            }

        messages = [
            Message.system(
                "Use the available tools to gather what you need. "
                "Call a tool only when you cannot answer from what you already have."
            ),
            Message.user(state["query"]),
        ]
        if state.get("plan"):
            messages.insert(1, Message.system(f"Your plan:\n{state['plan']}"))

        request = CompletionRequest(
            messages=tuple(messages),
            model=state["model"],
            max_tokens=1024,
            temperature=0.0,
            tools=deps.tools.specs(),
            tenant_id=state["tenant_id"],
            node="tool_executor",
        )
        try:
            completion = await deps.router.complete(request)
        except Exception as exc:  # noqa: BLE001 - answer from documents instead
            log.error("tool planning call failed", reason=str(exc))
            return {
                "node_trace": ["tool_executor"],
                "budget": budget,
                "events": [_event("tool_executor", "failed", started, detail=str(exc))],
            }

        budget = budget.spend_usage(completion.usage, cost_usd=completion.cost_usd)
        if not completion.tool_calls:
            return {
                "node_trace": ["tool_executor"],
                "budget": budget,
                "events": [_event("tool_executor", "no_tools_needed", started)],
            }

        calls = list(completion.tool_calls)[: budget.remaining_tool_calls]
        results = await _run_tools(deps, calls, tenant_id=state["tenant_id"])
        budget = budget.spend_tool_call(len(calls))

        return {
            "node_trace": ["tool_executor"],
            "tool_results": results,
            "pending_tool_calls": [],
            "budget": budget,
            "events": [
                _event(
                    "tool_executor",
                    "ok",
                    started,
                    tools=[c.name for c in calls],
                    failures=sum(1 for r in results if r.get("error")),
                )
            ],
        }

    return tool_executor


async def _run_tools(
    deps: NodeDependencies, calls: Sequence[ToolCall], *, tenant_id: str
) -> list[dict[str, Any]]:
    """Execute tool calls, turning failures into results rather than exceptions.

    A failed tool is information the model can act on ("that query was invalid,
    try another"), so it is returned as a result with an ``error`` field rather
    than propagated.
    """
    import asyncio

    if deps.tools is None:
        return []

    async def run_one(call: ToolCall) -> dict[str, Any]:
        try:
            output = await deps.tools.call(call.name, call.arguments, tenant_id=tenant_id)
        except Exception as exc:  # noqa: BLE001 - a tool failure is data, not a crash
            log.warning("tool call failed", tool=call.name, reason=str(exc))
            return {"tool": call.name, "call_id": call.id, "error": str(exc)}
        return {"tool": call.name, "call_id": call.id, "output": output}

    return list(await asyncio.gather(*(run_one(call) for call in calls)))


def make_generator_node(deps: NodeDependencies) -> Callable[[AgentState], Awaitable[dict]]:
    """Generate the cited answer from the retrieved context."""

    async def generator(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        chunks = context_chunks(state)
        budget = state["budget"].next_iteration()

        if budget.exhausted:
            return {
                "node_trace": ["generator"],
                "budget": budget,
                "stop_reason": StopReason.BUDGET_EXHAUSTED,
                "answer": _budget_message(),
                "events": [_event("generator", "budget_exhausted", started)],
            }

        prompt = deps.prompts.render(
            "answer",
            question=state["query"],
            context=format_context(chunks, tool_results=state.get("tool_results") or []),
            tenant_instructions=state.get("tenant_instructions"),
            response_template=state.get("response_template"),
            conversation_summary=state.get("conversation_summary"),
        )
        messages: list[Message] = [Message.system(prompt)]
        messages.extend(state.get("history") or [])
        messages.append(Message.user(state["query"]))
        if state.get("critique"):
            messages.append(
                Message.user(
                    "Revise your previous answer to address this review, keeping "
                    f"everything it did not object to:\n{state['critique']}"
                )
            )

        try:
            text, budget = await _complete(
                deps,
                {**state, "budget": budget},  # type: ignore[arg-type]
                messages=messages,
                node="generator",
                max_tokens=1500,
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001 - surface a real message, not a stack trace
            log.error("generation failed", reason=str(exc))
            return {
                "node_trace": ["generator"],
                "budget": budget,
                "stop_reason": StopReason.ERROR,
                "error": str(exc),
                "answer": "I could not generate an answer just now. Please try again.",
                "events": [_event("generator", "failed", started, detail=str(exc))],
            }

        return {
            "node_trace": ["generator"],
            "draft_answer": text,
            "answer": text,
            "budget": budget,
            "events": [_event("generator", "ok", started, chars=len(text))],
        }

    return generator


def make_citation_binder_node(deps: NodeDependencies) -> Callable[[AgentState], Awaitable[dict]]:
    """Verify each citation and drop the ones the cited passage does not support."""

    async def citation_binder(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        answer = state.get("answer")
        chunks = context_chunks(state)
        if not answer or not chunks or deps.citation_verifier is None:
            return {"node_trace": ["citation_binder"]}

        report = await deps.citation_verifier.verify(answer, chunks)
        citations = [
            {
                "marker": check.marker,
                "chunk_id": check.chunk_id,
                "claim": check.claim.text,
                "entailment": check.entailment.value,
                "score": round(check.score, 4),
                "verified": check.supported,
            }
            for check in report.checks
        ]
        return {
            "node_trace": ["citation_binder"],
            "answer": report.corrected_answer,
            "citations": citations,
            "events": [
                _event(
                    "citation_binder",
                    "ok",
                    started,
                    dropped=list(report.dropped_markers),
                    precision=round(report.citation_precision, 3),
                    recall=round(report.citation_recall, 3),
                )
            ],
        }

    return citation_binder


def make_self_critic_node(deps: NodeDependencies) -> Callable[[AgentState], Awaitable[dict]]:
    """Review the draft and decide whether it needs one more pass."""

    async def self_critic(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        answer = state.get("answer")
        if not answer or state.get("revision_count", 0) >= MAX_REVISIONS:
            return {"node_trace": ["self_critic"], "critique_verdict": "ACCEPT"}

        prompt = deps.prompts.render(
            "self_critic",
            question=state["query"],
            answer=answer,
            context=format_context(context_chunks(state)),
        )
        try:
            text, budget = await _complete(
                deps, state, messages=[Message.user(prompt)], node="self_critic", max_tokens=400
            )
        except Exception as exc:  # noqa: BLE001 - accept rather than block on a critic outage
            log.warning("self-critique unavailable; accepting the draft", reason=str(exc))
            return {
                "node_trace": ["self_critic"],
                "critique_verdict": "ACCEPT",
                "events": [_event("self_critic", "degraded", started, detail=str(exc))],
            }

        verdict = _parse_verdict(text)
        update: dict[str, Any] = {
            "node_trace": ["self_critic"],
            "critique_verdict": verdict,
            "budget": budget,
            "events": [_event("self_critic", "ok", started, verdict=verdict)],
        }
        if verdict == "REVISE":
            update["critique"] = text
            update["revision_count"] = state.get("revision_count", 0) + 1
        return update

    return self_critic


def make_output_guardrails_node(deps: NodeDependencies) -> Callable[[AgentState], Awaitable[dict]]:
    """Screen the finished answer before it reaches the user."""

    async def output_guardrails(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        answer = state.get("answer")
        if not answer or deps.output_guardrails is None:
            return {"node_trace": ["output_guardrails"]}

        outcome = await deps.output_guardrails.run(
            answer,
            context=deps.guardrail_context(
                state,
                retrieved_chunks=context_chunks(state),
                top_retrieval_score=state.get("top_retrieval_score"),
            ),
        )
        update: dict[str, Any] = {
            "node_trace": ["output_guardrails"],
            "output_guardrails": list(outcome.results),
            "events": [_event("output_guardrails", "ok", started, flags=list(outcome.flags))],
        }
        if outcome.blocked:
            blocking = outcome.blocking_result
            update["answer"] = _refusal_for(blocking.kind.value if blocking else "policy")
            update["blocked_reason"] = blocking.reason if blocking else "blocked by policy"
            update["stop_reason"] = StopReason.GUARDRAIL_BLOCKED
        elif outcome.was_modified:
            update["answer"] = outcome.text
        return update

    return output_guardrails


def make_formatter_node(deps: NodeDependencies) -> Callable[[AgentState], Awaitable[dict]]:
    """Apply the tenant's response template and settle the stop reason."""

    async def formatter(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        answer = state.get("answer") or ""
        template = state.get("response_template")

        if template and answer:
            from jinja2 import Environment, StrictUndefined

            try:
                answer = (
                    Environment(undefined=StrictUndefined, autoescape=False)  # noqa: S701
                    .from_string(template)
                    .render(answer=answer, citations=state.get("citations") or [])
                )
            except Exception as exc:  # noqa: BLE001 - a bad template must not lose the answer
                log.warning(
                    "tenant response template failed; using the raw answer", reason=str(exc)
                )

        return {
            "node_trace": ["formatter"],
            "answer": answer,
            "stop_reason": state.get("stop_reason") or StopReason.COMPLETED,
            "events": [_event("formatter", "ok", started)],
        }

    return formatter


# ── Helpers ──────────────────────────────────────────────────────────────────


def format_context(
    chunks: Sequence[RetrievedChunk], *, tool_results: Sequence[dict[str, Any]] = ()
) -> str:
    """Render retrieved chunks as numbered, citable context.

    The numbering is 1-based and defines what ``[n]`` means for the rest of the
    turn, so the citation binder resolves markers against exactly this order.

    Example:
        >>> from src.retrieval.types import RetrievalSource
        >>> chunk = RetrievedChunk(
        ...     chunk_id="c1", content="Thirty days.", score=1.0,
        ...     source=RetrievalSource.DENSE, document_title="Policy",
        ... )
        >>> print(format_context([chunk]))
        [1] Policy
        Thirty days.
    """
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        header = chunk.document_title or chunk.chunk_id
        if chunk.page_number is not None:
            header = f"{header}, p. {chunk.page_number}"
        blocks.append(f"[{index}] {header}\n{chunk.content}")

    for result in tool_results:
        label = result.get("tool", "tool")
        if result.get("error"):
            blocks.append(f"[tool:{label}] failed: {result['error']}")
        else:
            blocks.append(f"[tool:{label}] {json.dumps(result.get('output'), default=str)[:2000]}")

    return "\n\n".join(blocks) if blocks else "(no context retrieved)"


def _parse_verdict(text: str) -> str:
    """Read the critic's verdict from its first line.

    Defaults to ACCEPT: an unparseable critique is not evidence of a defect, and
    treating it as one would send every turn into a revision loop.

    Example:
        >>> _parse_verdict(chr(10).join(["REVISE", "unsupported: ..."]))
        'REVISE'
        >>> _parse_verdict("The answer looks fine to me")
        'ACCEPT'
    """
    first = text.strip().splitlines()[0].strip().upper() if text.strip() else ""
    for verdict in ("REJECT", "REVISE", "ACCEPT"):
        if first.startswith(verdict):
            return verdict
    return "ACCEPT"


def _refusal_for(kind: str) -> str:
    """A user-facing message for a blocked turn.

    Says what happened without echoing the offending content or naming the
    detector, which would both leak information and teach an attacker what to
    change.

    Example:
        >>> _refusal_for("pii").startswith("I could not")
        True
    """
    messages = {
        "prompt_injection": (
            "I could not process that request because it appears to contain "
            "instructions intended to change how I work."
        ),
        "pii": (
            "I could not process that request because it contains personal data "
            "that this workspace is configured to block."
        ),
        "toxicity": "I could not process that request.",
        "moderation": "I could not share that answer.",
        "hallucination": (
            "I could not produce an answer I can support from your documents, so "
            "I would rather not answer than guess."
        ),
    }
    return messages.get(kind, "I could not complete that request.")


def _budget_message() -> str:
    """Message shown when a turn runs out of budget mid-flight."""
    return (
        "I ran out of the budget allocated for this turn before I could finish. "
        "Try asking a narrower question, or split it into parts."
    )
