"""Dependency wiring.

Expensive objects — embedders, rerankers, provider clients, the compiled graph —
are built once at startup and held on ``app.state``. Building them per request
would load a 400MB cross-encoder on every message.

Tenant-specific configuration is resolved *per request* on top of those shared
objects, because two tenants can run different strategies, models, prompts and
guardrail policies against the same process. :class:`TenantRuntime` is that
per-request assembly: cheap to build, and it holds no I/O of its own.

Example:
    >>> from src.api.dependencies import build_model_policy
    >>> build_model_policy({}, default="claude-sonnet-5").default_model
    'claude-sonnet-5'
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, Any

from fastapi import Depends, Request

from src.core.config import Settings, get_settings
from src.core.logging import get_logger
from src.guardrails.base import GuardrailPolicy
from src.services.llm.router import LLMRouter, ModelPolicy, RetryConfig

log = get_logger(__name__)


@dataclass(slots=True)
class AppServices:
    """Process-wide singletons, built once at startup."""

    settings: Settings
    embedder: Any
    reranker: Any
    prompts: Any
    providers: dict[str, Any]
    cache: Any
    semantic_cache: Any
    tool_cache: Any
    tools: Any
    rate_limiter: Any
    budget_tracker: Any
    redis: Any = None
    opensearch: Any = None
    neo4j: Any = None
    storage: Any = None
    langfuse: Any = None


def build_model_policy(raw: dict[str, Any], *, default: str) -> ModelPolicy:
    """Build a tenant's model policy from its stored configuration.

    Unknown keys are ignored and missing ones fall back, so a policy written by a
    newer deployment does not break an older worker mid-rollout.

    Example:
        >>> policy = build_model_policy({"allowed_models": ["gpt-4o"]}, default="gpt-4o")
        >>> policy.allows("claude-sonnet-5")
        False
    """
    return ModelPolicy(
        default_model=str(raw.get("default_model") or default),
        cheap_model=str(raw.get("cheap_model") or "claude-haiku-4-5-20251001"),
        allowed_models=tuple(raw.get("allowed_models") or ()),
        fallback_models=tuple(raw.get("fallback_models") or ()),
        max_tokens_per_request=int(raw.get("max_tokens_per_request") or 16_000),
        cost_aware_routing=bool(raw.get("cost_aware_routing", True)),
    )


def build_guardrail_policy(raw: dict[str, Any]) -> GuardrailPolicy:
    """Build a tenant's guardrail policy from its stored configuration.

    Validation failures fall back to the defaults rather than refusing the
    request: a malformed policy row should degrade to the safe standard
    behaviour, not take the workspace offline.

    Example:
        >>> build_guardrail_policy({"pii_mode": "block"}).pii_mode
        'block'
        >>> build_guardrail_policy({"pii_threshold": "not a number"}).pii_enabled
        True
    """
    try:
        return GuardrailPolicy.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001 - fall back to safe defaults
        log.warning("invalid guardrail config; using defaults", reason=str(exc))
        return GuardrailPolicy()


@dataclass(slots=True)
class TenantRuntime:
    """Everything one request needs, resolved for its tenant."""

    tenant: Any
    router: LLMRouter
    retriever: Any
    guardrail_policy: GuardrailPolicy
    agent: Any
    model: str
    services: AppServices

    @property
    def tenant_id(self) -> str:
        """The active tenant's id."""
        return str(self.tenant.id)


def get_services(request: Request) -> AppServices:
    """Return the process-wide services.

    Raises:
        RuntimeError: when startup has not completed, which would otherwise
            surface as a confusing AttributeError deep in a route.
    """
    services: AppServices | None = getattr(request.app.state, "services", None)
    if services is None:  # pragma: no cover - only reachable mid-startup
        msg = "application services are not initialised"
        raise RuntimeError(msg)
    return services


async def get_tenant_runtime(
    request: Request,
    services: Annotated[AppServices, Depends(get_services)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TenantRuntime:
    """Assemble the per-tenant runtime for this request."""
    return await build_tenant_runtime(request.state.principal, services=services, settings=settings)


async def build_tenant_runtime(
    principal: Any, *, services: AppServices, settings: Settings
) -> TenantRuntime:
    """Assemble a per-tenant runtime.

    Separated from the FastAPI dependency so the batch worker can build the same
    runtime without inventing a request. Two assembly paths would drift, and the
    one that drifts is always the one nobody watches — meaning batch answers
    would quietly stop matching interactive ones.

    The tenant row is loaded once here and threaded through, rather than being
    re-read by each component that needs a setting from it.
    """
    from src.agents.graph import AgentRunner
    from src.agents.nodes import NodeDependencies
    from src.api.auth import load_tenant
    from src.core.db import session_scope
    from src.guardrails.base import GuardrailPipeline
    from src.guardrails.content import ModerationGuardrail, OffTopicGuardrail
    from src.guardrails.groundedness import CitationVerifier, GroundednessGuardrail
    from src.guardrails.injection import InjectionGuardrail
    from src.guardrails.pii import PiiGuardrail
    from src.models.telemetry import GuardrailStage

    async with session_scope() as session:
        tenant = await load_tenant(session, principal.tenant_id)

    guardrail_policy = build_guardrail_policy(tenant.guardrail_config or {})
    model_policy = build_model_policy(
        tenant.model_policy or {}, default=settings.default_chat_model
    )

    budget = await services.budget_tracker.status(
        tenant.id,
        day=_today(),
        tokens_limit=tenant.daily_token_budget,
        cost_limit_usd=float(tenant.monthly_cost_cap_usd) if tenant.monthly_cost_cap_usd else None,
    )
    budget.raise_if_exhausted()

    from src.observability.metrics import set_budget_used

    set_budget_used(tenant_id=tenant.id, ratio=budget.fraction_used)

    router = LLMRouter(
        providers=services.providers,
        policy=model_policy,
        retry=RetryConfig(max_attempts=settings.llm_max_retries),
        budget_remaining_tokens=budget.tokens_remaining,
        usage_sink=_usage_sink(tenant.id),
    )

    retriever = await _build_retriever(
        tenant=tenant, services=services, router=router, settings=settings
    )

    deps = NodeDependencies(
        router=router,
        prompts=services.prompts,
        retriever=retriever,
        tools=services.tools,
        input_guardrails=GuardrailPipeline(
            [
                InjectionGuardrail(router=router, judge_model=model_policy.cheap_model),
                PiiGuardrail(stage=GuardrailStage.INPUT),
                OffTopicGuardrail(),
            ],
            stage=GuardrailStage.INPUT,
        ),
        output_guardrails=GuardrailPipeline(
            [
                GroundednessGuardrail(verifier=CitationVerifier()),
                PiiGuardrail(stage=GuardrailStage.OUTPUT),
                ModerationGuardrail(
                    api_key=(
                        settings.openai_api_key.get_secret_value()
                        if settings.openai_api_key
                        else None
                    )
                ),
            ],
            stage=GuardrailStage.OUTPUT,
        ),
        citation_verifier=CitationVerifier(),
        semantic_cache=services.semantic_cache,
        policy=guardrail_policy,
        top_k=settings.retrieval_top_k,
    )

    return TenantRuntime(
        tenant=tenant,
        router=router,
        retriever=retriever,
        guardrail_policy=guardrail_policy,
        agent=AgentRunner(deps),
        model=model_policy.default_model,
        services=services,
    )


async def _build_retriever(
    *, tenant: Any, services: AppServices, router: LLMRouter, settings: Settings
) -> Any:
    """Assemble the hybrid retriever for one tenant's enabled strategies."""
    from src.core.db import session_scope
    from src.retrieval.corrective import NoWebSearch, RetrievalEvaluator, TavilyWebSearch
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.graph import GraphRetriever
    from src.retrieval.hybrid import HybridConfig, HybridRetriever
    from src.retrieval.rewrite import QueryRewriter
    from src.retrieval.sparse import SparseRetriever

    config = HybridConfig.from_strategies(
        tenant.enabled_strategies or ["hybrid"],
        top_k=settings.retrieval_top_k,
        expanded_k=settings.retrieval_expanded_k,
        rerank_top_n=settings.rerank_top_n,
    )

    backends: list[Any] = []
    # The dense retriever needs a live session, so it is constructed per request
    # around a session that lives for the request's duration.
    session_context = session_scope()
    session = await session_context.__aenter__()
    backends.append(DenseRetriever(session=session, embedder=services.embedder))

    if config.use_sparse and services.opensearch is not None:
        backends.append(SparseRetriever(client=services.opensearch))
    if config.use_graph and services.neo4j is not None:
        backends.append(GraphRetriever(driver=services.neo4j, router=router))

    web_search = (
        TavilyWebSearch(api_key=settings.tavily_api_key.get_secret_value())
        if settings.tavily_api_key
        else NoWebSearch()
    )

    return HybridRetriever(
        retrievers=backends,
        config=config,
        reranker=services.reranker,
        rewriter=QueryRewriter(router=router),
        evaluator=RetrievalEvaluator(router=router) if config.use_corrective else None,
        web_search=web_search,
    )


def _usage_sink(tenant_id: str) -> Callable[[Any, Any], Awaitable[None]]:
    """Build the callback that persists usage and updates metrics.

    Failures here are swallowed: losing an accounting row is bad, and failing a
    user's answer because accounting failed is worse. The durable counter and
    the Redis counter disagreeing is recoverable; a failed turn is not.
    """

    async def sink(completion: Any, request: Any) -> None:
        from src.observability.metrics import record_cost, record_provider_failure

        record_cost(
            tenant_id=tenant_id,
            model=completion.model,
            operation=request.node or "chat",
            cost_usd=completion.cost_usd,
        )
        if completion.was_fallback:
            record_provider_failure(provider=completion.provider, recovered=True)

        try:
            from src.repositories.usage import record_usage

            await record_usage(tenant_id=tenant_id, completion=completion, request=request)
        except Exception as exc:  # noqa: BLE001 - accounting must not fail a turn
            log.warning("failed to persist usage record", reason=str(exc))

    return sink


def _today() -> str:
    """Today's date in UTC, as the budget counters key it."""
    from datetime import UTC, datetime

    return datetime.now(UTC).date().isoformat()


#: Services are expensive to build (models, clients, pools) and the worker
#: process runs many batches, so the bundle is built once and reused.
_worker_services: AppServices | None = None


async def build_batch_runtime(tenant_id: str) -> tuple[TenantRuntime, Any]:
    """Build a runtime and a principal for a batch item in the worker.

    The principal carries the tenant and no user, because a batch is submitted
    by an API key rather than a person, and inventing a user id would put a name
    on usage records that nobody actually attached to.
    """
    global _worker_services

    from src.core.config import get_settings as load_settings
    from src.main import build_services

    settings = load_settings()
    if _worker_services is None:
        _worker_services = await build_services(settings)

    principal = SimpleNamespace(
        tenant_id=tenant_id,
        user_id=None,
        api_key_id=None,
        is_admin=False,
        method="batch",
        scopes=frozenset({"read", "write"}),
        has_scope=lambda _scope: True,
        require=lambda _scope: None,
    )
    runtime = await build_tenant_runtime(principal, services=_worker_services, settings=settings)
    return runtime, principal
