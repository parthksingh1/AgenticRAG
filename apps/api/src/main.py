"""FastAPI application entry point.

Startup builds the expensive singletons once — models, provider clients, the
prompt registry, the tool registry — and puts them on ``app.state``. A request
never constructs any of them.

Startup is deliberately tolerant. Every optional dependency is attempted, and a
failure is logged and recorded rather than aborting the boot. A process that
refuses to start because OpenSearch is not up yet is a process that cannot serve
the dense-only traffic it is perfectly capable of serving, and it turns a
degraded dependency into a full outage. What is genuinely required — the
database and Redis — is enforced by ``/readyz`` instead, so the pod starts,
reports itself unready, and joins the load balancer when its dependencies
arrive.

Example:
    >>> from src.main import create_app
    >>> app = create_app()
    >>> app.title
    'AgenticRAG API'
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from src.core.config import Settings, get_settings
from src.core.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build shared services on startup and release them on shutdown."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.is_production)
    log.info(
        "starting AgenticRAG API",
        env=settings.app_env.value,
        providers=settings.configured_providers,
    )

    from src.observability.tracing import setup_langfuse, setup_tracing

    setup_tracing(settings, app)
    langfuse = setup_langfuse(settings)

    services = await build_services(settings, langfuse=langfuse)
    app.state.services = services
    app.state.rate_limiter = services.rate_limiter
    app.state.settings = settings

    from src.api.auth import ClerkVerifier

    app.state.clerk = ClerkVerifier(jwks_url=settings.clerk_jwks_url, issuer=settings.clerk_issuer)

    log.info("startup complete", tools=len(services.tools.names()) if services.tools else 0)
    try:
        yield
    finally:
        await shutdown_services(services)


async def build_services(settings: Settings, *, langfuse: Any = None) -> Any:
    """Construct the process-wide services.

    Each optional dependency is attempted independently. A failure disables the
    feature that needs it, and ``/readyz`` reports the gap.
    """
    from src.api.dependencies import AppServices
    from src.caching.base import InMemoryCache, RedisCache, ToolResultCache
    from src.caching.semantic import SemanticCache
    from src.guardrails.limits import BudgetTracker, InMemoryTokenBucket, TokenBucket
    from src.ingestion.embedders.base import (
        CachingEmbedder,
        HashingEmbedder,
        InMemoryEmbeddingCache,
        SentenceTransformerEmbedder,
    )
    from src.mcp_clients.registry import DEFAULT_SERVERS, build_registry
    from src.retrieval.rerank import CrossEncoderReranker, IdentityReranker
    from src.services.prompts import get_prompt_registry

    prompts = get_prompt_registry(settings.prompts_dir)

    embedder = _optional(
        "embedder",
        lambda: CachingEmbedder(
            SentenceTransformerEmbedder(settings.default_embedding_model),
            cache=InMemoryEmbeddingCache(),
        ),
        # The hashing embedder keeps the API serving when the model cannot be
        # downloaded. Retrieval quality collapses, so this is logged loudly and
        # surfaced as degraded rather than passing silently.
        fallback=lambda: CachingEmbedder(
            HashingEmbedder(dimension=settings.embedding_dim), cache=InMemoryEmbeddingCache()
        ),
    )
    reranker = _optional(
        "reranker",
        lambda: CrossEncoderReranker(settings.default_reranker_model),
        fallback=IdentityReranker,
    )

    redis = await _optional_async("redis", lambda: _build_redis(settings))
    opensearch = await _optional_async("opensearch", lambda: _build_opensearch(settings))
    neo4j = await _optional_async("neo4j", lambda: _build_neo4j(settings))
    storage = _optional("storage", lambda: _build_storage(settings), fallback=lambda: None)

    cache = RedisCache(redis=redis) if redis is not None else InMemoryCache()
    providers = _build_providers(settings)

    tools = build_registry(DEFAULT_SERVERS, cache=ToolResultCache(cache))
    try:
        await tools.discover()
    except Exception as exc:  # noqa: BLE001 - tools are optional
        log.warning("MCP discovery failed; tools disabled for now", reason=str(exc))

    return AppServices(
        settings=settings,
        embedder=embedder,
        reranker=reranker,
        prompts=prompts,
        providers=providers,
        cache=cache,
        semantic_cache=SemanticCache(
            cache=cache, embedder=embedder, threshold=settings.semantic_cache_threshold
        ),
        tool_cache=ToolResultCache(cache),
        tools=tools,
        rate_limiter=(
            TokenBucket(
                redis=redis,
                capacity=settings.rate_limit_chat_per_minute,
                refill_per_second=settings.rate_limit_chat_per_minute / 60,
            )
            if redis is not None
            else InMemoryTokenBucket(
                capacity=settings.rate_limit_chat_per_minute,
                refill_per_second=settings.rate_limit_chat_per_minute / 60,
            )
        ),
        budget_tracker=BudgetTracker(redis=redis) if redis is not None else _NullBudgetTracker(),
        redis=redis,
        opensearch=opensearch,
        neo4j=neo4j,
        storage=storage,
        langfuse=langfuse,
    )


def _build_providers(settings: Settings) -> dict[str, Any]:
    """Build a client for every provider that has a key configured.

    A provider without a key is simply absent, and the router reports "provider
    not configured" if a model routes to it — which is a clearer failure than a
    401 from upstream.
    """
    from src.services.llm.providers import (
        AnthropicProvider,
        GoogleProvider,
        OpenAICompatibleProvider,
    )

    providers: dict[str, Any] = {}
    timeout = settings.llm_timeout_seconds

    if settings.anthropic_api_key:
        providers["anthropic"] = _optional(
            "anthropic",
            lambda: AnthropicProvider(
                api_key=settings.anthropic_api_key.get_secret_value(), timeout=timeout
            ),
            fallback=lambda: None,
        )
    if settings.openai_api_key:
        providers["openai"] = _optional(
            "openai",
            lambda: OpenAICompatibleProvider(
                api_key=settings.openai_api_key.get_secret_value(), timeout=timeout
            ),
            fallback=lambda: None,
        )
    if settings.google_api_key:
        providers["google"] = _optional(
            "google",
            lambda: GoogleProvider(
                api_key=settings.google_api_key.get_secret_value(), timeout=timeout
            ),
            fallback=lambda: None,
        )
    if settings.groq_api_key:
        providers["groq"] = _optional(
            "groq",
            lambda: OpenAICompatibleProvider(
                api_key=settings.groq_api_key.get_secret_value(),
                name="groq",
                base_url="https://api.groq.com/openai/v1",
                timeout=timeout,
            ),
            fallback=lambda: None,
        )
    if settings.together_api_key:
        providers["together"] = _optional(
            "together",
            lambda: OpenAICompatibleProvider(
                api_key=settings.together_api_key.get_secret_value(),
                name="together",
                base_url="https://api.together.xyz/v1",
                timeout=timeout,
            ),
            fallback=lambda: None,
        )

    live = {name: client for name, client in providers.items() if client is not None}
    if not live:
        log.warning("no LLM providers configured; chat will fail until a key is set in .env")
    return live


async def _build_redis(settings: Settings) -> Any:
    """Connect to Redis, verifying the connection before returning it."""
    import redis.asyncio as redis_async

    client = redis_async.from_url(settings.redis_url, decode_responses=False)
    await client.ping()
    return client


async def _build_opensearch(settings: Settings) -> Any:
    """Connect to OpenSearch, verifying the cluster answers."""
    from opensearchpy import AsyncOpenSearch

    auth = None
    if settings.opensearch_user and settings.opensearch_password:
        auth = (settings.opensearch_user, settings.opensearch_password.get_secret_value())

    client = AsyncOpenSearch(hosts=[settings.opensearch_url], http_auth=auth, timeout=10)
    await client.info()
    return client


async def _build_neo4j(settings: Settings) -> Any:
    """Connect to Neo4j, verifying connectivity."""
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
    await driver.verify_connectivity()
    return driver


def _build_storage(settings: Settings) -> Any:
    """Build the object storage client."""
    from src.services.storage import ObjectStorage

    return ObjectStorage(settings)


def _optional(name: str, build: Any, *, fallback: Any) -> Any:
    """Build a component, falling back when it cannot be constructed."""
    try:
        return build()
    except Exception as exc:  # noqa: BLE001 - a degraded feature, not a dead process
        log.warning("component unavailable; using the fallback", component=name, reason=str(exc))
        return fallback()


async def _optional_async(name: str, build: Any) -> Any:
    """Connect to an optional dependency, returning None when it is unreachable."""
    try:
        return await build()
    except Exception as exc:  # noqa: BLE001 - reported by /readyz instead
        log.warning("dependency unavailable at startup", dependency=name, reason=str(exc))
        return None


class _NullBudgetTracker:
    """Budget tracker used when Redis is unavailable.

    Reports zero usage so the API keeps serving. The durable counter still
    records real spend, so nothing is lost — but enforcement is degraded until
    Redis returns, which ``/readyz`` reports.
    """

    async def record(self, *_args: Any, **_kwargs: Any) -> None:
        """Discard the record; the database write still happens elsewhere."""
        return

    async def status(self, _tenant_id: str, **kwargs: Any) -> Any:
        """Report an unspent budget."""
        from src.guardrails.limits import BudgetStatus

        return BudgetStatus(
            tokens_used=0,
            tokens_limit=int(kwargs.get("tokens_limit", 0)),
            cost_usd=0.0,
            cost_limit_usd=kwargs.get("cost_limit_usd"),
        )


async def shutdown_services(services: Any) -> None:
    """Release everything the process holds.

    Each close is independent: one that fails must not prevent the rest, or a
    stuck client would leak every other connection on every deploy.
    """
    from src.core.db import dispose_engine
    from src.observability.tracing import shutdown_tracing

    closers = (
        ("tools", getattr(services.tools, "aclose", None)),
        ("reranker", getattr(services.reranker, "aclose", None)),
        ("neo4j", getattr(services.neo4j, "close", None)),
        ("opensearch", getattr(services.opensearch, "close", None)),
        ("redis", getattr(services.redis, "aclose", None)),
    )
    for name, close in closers:
        if close is None:
            continue
        try:
            await close()
        except Exception as exc:  # noqa: BLE001 - shutdown is best effort
            log.warning("failed to close cleanly", component=name, reason=str(exc))

    for provider in services.providers.values():
        try:
            await provider.aclose()
        except Exception as exc:  # noqa: BLE001 - shutdown is best effort
            log.warning("failed to close provider", reason=str(exc))

    await shutdown_tracing()
    await dispose_engine()
    log.info("shutdown complete")


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="AgenticRAG API",
        version="0.1.0",
        description=(
            "Multi-tenant agentic RAG. The `/v1/chat/completions` endpoint is "
            "OpenAI-compatible, so any OpenAI SDK works against it unchanged."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    from src.api.middleware import install_middleware

    install_middleware(app, settings=settings)

    from src.api.routers import chat, health

    app.include_router(health.router)
    app.include_router(chat.router)

    for module_name in ("documents", "search", "admin", "openai_compat", "batch"):
        _include_optional(app, module_name)

    return app


def _include_optional(app: FastAPI, module_name: str) -> None:
    """Include a router if its module is present.

    Lets the router set grow without this file needing to change, and keeps a
    partially-built deployment serving what it does have.
    """
    import importlib

    try:
        module = importlib.import_module(f"src.api.routers.{module_name}")
    except ImportError as exc:
        log.debug("router not available", router=module_name, reason=str(exc))
        return
    app.include_router(module.router)


app = create_app()
