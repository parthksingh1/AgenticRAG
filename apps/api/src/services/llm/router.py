"""Model routing, retries and cross-provider fallback.

Three responsibilities, deliberately in one place so they compose correctly:

* **Retry** — exponential backoff with full jitter on transient failures. Jitter
  matters: without it, a provider blip synchronises every worker into a
  thundering herd that reproduces the outage.
* **Fallback** — when the primary provider is still failing after its retries,
  the same request is replayed against a different provider. The response is
  marked ``was_fallback`` so the cost dashboard and evals can tell the
  difference rather than quietly comparing apples to oranges.
* **Cost-aware selection** — a cheap classifier decides whether a query needs a
  frontier model at all. Most questions do not, and routing them to a mini model
  is the single largest cost lever in the system.

Every call emits a usage record, so there is no path that spends money without
attribution.

Example:
    >>> policy = ModelPolicy(default_model="claude-sonnet-5")
    >>> policy.allows("claude-sonnet-5")
    True
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from src.core.errors import BudgetExceededError, ProviderError
from src.core.logging import get_logger
from src.services.llm.pricing import context_window, provider_for, supports
from src.services.llm.providers import Provider
from src.services.llm.types import (
    Completion,
    CompletionRequest,
    Message,
    StreamEvent,
    StreamEventType,
    Usage,
)

log = get_logger(__name__)

#: Called after every provider call with the completion, so the service layer
#: can persist a UsageRecord without the router importing the database.
UsageSink = Callable[[Completion, CompletionRequest], Awaitable[None]]


class ModelPolicy(BaseModel):
    """Which models a tenant may use, and how spend is capped."""

    model_config = ConfigDict(frozen=True)

    default_model: str
    cheap_model: str = "claude-haiku-4-5-20251001"
    #: Empty means "any registered model". Non-empty is an allowlist.
    allowed_models: tuple[str, ...] = ()
    #: Ordered providers to try when the primary fails. Empty disables fallback.
    fallback_models: tuple[str, ...] = ()
    max_tokens_per_request: int = Field(default=16_000, ge=1)
    cost_aware_routing: bool = True

    def allows(self, model: str) -> bool:
        """Whether the tenant may use a model.

        Example:
            >>> ModelPolicy(default_model="a", allowed_models=("a",)).allows("b")
            False
        """
        return not self.allowed_models or model in self.allowed_models

    def resolve(self, requested: str | None) -> str:
        """Pick the model to use, falling back to the default when disallowed.

        Refusing outright would break a tenant whose allowlist was tightened
        after a conversation started, so an unavailable model degrades to the
        tenant's default and is logged.

        Example:
            >>> ModelPolicy(default_model="a", allowed_models=("a",)).resolve("b")
            'a'
        """
        if requested and self.allows(requested):
            return requested
        if requested:
            log.warning(
                "requested model not permitted for tenant; using default",
                requested=requested,
                default=self.default_model,
            )
        return self.default_model


class RetryConfig(BaseModel):
    """Backoff parameters for transient provider failures."""

    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=4, ge=1, le=10)
    initial_delay_seconds: float = Field(default=0.5, gt=0)
    max_delay_seconds: float = Field(default=8.0, gt=0)
    multiplier: float = Field(default=2.0, gt=1.0)

    def delay_for(self, attempt: int, *, jitter: float = 1.0) -> float:
        """Seconds to wait before ``attempt`` (1-based), with full jitter.

        Args:
            attempt: The attempt about to be made, counting from 1.
            jitter: Fraction of the computed ceiling to actually wait. Injected
                so the tests are deterministic rather than sleeping randomly.

        Example:
            >>> RetryConfig().delay_for(1, jitter=1.0)
            0.5
            >>> RetryConfig().delay_for(3, jitter=1.0)
            2.0
            >>> RetryConfig().delay_for(99, jitter=1.0)
            8.0
        """
        ceiling = min(
            self.initial_delay_seconds * (self.multiplier ** (attempt - 1)),
            self.max_delay_seconds,
        )
        return ceiling * jitter


@dataclass(slots=True)
class RouterStats:
    """Counters exposed to the metrics layer."""

    calls: int = 0
    retries: int = 0
    fallbacks: int = 0
    failures: int = 0
    downgrades: int = 0
    cost_usd: float = 0.0
    usage: Usage = field(default_factory=Usage)


class LLMRouter:
    """Routes completion requests to providers with retry and fallback."""

    def __init__(
        self,
        *,
        providers: dict[str, Provider],
        policy: ModelPolicy,
        retry: RetryConfig | None = None,
        usage_sink: UsageSink | None = None,
        budget_remaining_tokens: int | None = None,
    ) -> None:
        """Create a router.

        Args:
            providers: Provider name -> adapter. Must cover every model the
                policy can select.
            policy: Tenant model policy.
            retry: Backoff configuration.
            usage_sink: Awaited after each call for persistence.
            budget_remaining_tokens: Hard ceiling for this router's lifetime.
                ``None`` disables the check (used by the eval harness).
        """
        self._providers = providers
        self._policy = policy
        self._retry = retry or RetryConfig()
        self._usage_sink = usage_sink
        self._budget_remaining = budget_remaining_tokens
        self.stats = RouterStats()

    async def complete(
        self,
        request: CompletionRequest,
        *,
        allow_fallback: bool = True,
    ) -> Completion:
        """Complete a request, retrying and falling back as configured.

        Raises:
            BudgetExceededError: when the tenant's token budget is exhausted.
            ProviderError: when every model in the chain has failed.
        """
        self._check_budget(request)
        chain = self._model_chain(request.model, allow_fallback=allow_fallback)

        last_error: Exception | None = None
        for position, model in enumerate(chain):
            try:
                completion = await self._complete_with_retry(
                    request.model_copy(update={"model": model})
                )
            except ProviderError as exc:
                last_error = exc
                log.warning(
                    "model failed after retries",
                    model=model,
                    remaining_fallbacks=len(chain) - position - 1,
                    reason=exc.details.get("reason"),
                )
                continue

            if position > 0:
                self.stats.fallbacks += 1
                completion = completion.model_copy(update={"was_fallback": True})
            await self._record(completion, request)
            return completion

        self.stats.failures += 1
        raise ProviderError(
            provider="all",
            reason=f"every model in {chain} failed; last error: {last_error}",
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """Stream a completion.

        Streaming deliberately does not fall back mid-stream: once tokens have
        reached the user, silently switching models would splice two different
        answers together. A failure before the first token is retried; a failure
        after it surfaces as an error event.
        """
        self._check_budget(request)
        model = self._policy.resolve(request.model)
        provider = self._provider_for(model)
        emitted_any = False

        async for event in provider.stream(request.model_copy(update={"model": model})):
            if event.type is StreamEventType.TEXT:
                emitted_any = True
            if event.type is StreamEventType.ERROR and not emitted_any:
                log.warning("stream failed before first token; retrying once", model=model)
                async for retry_event in provider.stream(
                    request.model_copy(update={"model": model})
                ):
                    yield retry_event
                return
            yield event

    async def classify_complexity(self, query: str, *, classifier_model: str | None = None) -> bool:
        """Decide whether a query needs the frontier model.

        Uses the cheap model to make the call, so the routing decision costs
        roughly 1% of the saving it produces. Any ambiguity resolves to "yes,
        use the frontier model": under-serving a hard question is a worse
        failure than over-paying for an easy one.

        Returns:
            True when the frontier model should be used.
        """
        if not self._policy.cost_aware_routing:
            return True

        prompt = (
            "Classify whether the question needs a frontier reasoning model.\n"
            "Answer SIMPLE for lookups, definitions and single-fact questions.\n"
            "Answer COMPLEX for multi-step reasoning, comparison, synthesis or "
            "ambiguous questions.\n"
            "Reply with exactly one word.\n\n"
            f"Question: {query}"
        )
        request = CompletionRequest(
            messages=(Message.user(prompt),),
            model=classifier_model or self._policy.cheap_model,
            max_tokens=5,
            temperature=0.0,
            node="cost_router",
        )
        try:
            completion = await self.complete(request, allow_fallback=False)
        except ProviderError:
            return True

        needs_frontier = "SIMPLE" not in completion.content.upper()
        if not needs_frontier:
            self.stats.downgrades += 1
        return needs_frontier

    async def select_model(self, query: str, *, requested: str | None = None) -> str:
        """Choose the model for a query, honouring policy and cost routing."""
        if requested:
            return self._policy.resolve(requested)
        if not self._policy.cost_aware_routing:
            return self._policy.default_model
        needs_frontier = await self.classify_complexity(query)
        chosen = self._policy.default_model if needs_frontier else self._policy.cheap_model
        return self._policy.resolve(chosen)

    async def aclose(self) -> None:
        """Close every provider client."""
        await asyncio.gather(*(p.aclose() for p in self._providers.values()))

    # ── internals ────────────────────────────────────────────────────────────

    async def _complete_with_retry(self, request: CompletionRequest) -> Completion:
        """Call one model, retrying transient failures with jittered backoff."""
        provider = self._provider_for(request.model)
        last_error: Exception | None = None

        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                self.stats.calls += 1
                return await provider.complete(request)
            except ProviderError as exc:
                last_error = exc
                if attempt == self._retry.max_attempts or not _is_retryable(exc):
                    break
                self.stats.retries += 1
                delay = self._retry.delay_for(attempt, jitter=random.random())  # noqa: S311
                log.info(
                    "retrying provider call",
                    provider=provider.name,
                    model=request.model,
                    attempt=attempt,
                    delay_seconds=round(delay, 3),
                )
                await asyncio.sleep(delay)

        raise ProviderError(
            provider=provider.name, reason=f"exhausted retries: {last_error}"
        ) from last_error

    def _model_chain(self, requested: str, *, allow_fallback: bool) -> list[str]:
        """The ordered list of models to try for one request."""
        primary = self._policy.resolve(requested)
        if not allow_fallback:
            return [primary]
        chain = [primary]
        chain.extend(
            model
            for model in self._policy.fallback_models
            if model != primary and self._policy.allows(model)
        )
        return chain

    def _provider_for(self, model: str) -> Provider:
        """Look up the adapter serving a model.

        Raises:
            ProviderError: when the model's provider is not configured, which
                means an API key is missing rather than a transient fault.
        """
        name = provider_for(model)
        provider = self._providers.get(name)
        if provider is None:
            raise ProviderError(
                provider=name, reason=f"provider not configured (missing API key?) for {model!r}"
            )
        return provider

    def _check_budget(self, request: CompletionRequest) -> None:
        """Fail before spending when the tenant has no budget left."""
        if self._budget_remaining is None:
            return
        if self._budget_remaining <= 0:
            raise BudgetExceededError(
                limit=0, used=self.stats.usage.total_tokens, window="router_lifetime"
            )
        if request.max_tokens > self._policy.max_tokens_per_request:
            raise BudgetExceededError(
                limit=self._policy.max_tokens_per_request,
                used=request.max_tokens,
                window="per_request",
            )

    async def _record(self, completion: Completion, request: CompletionRequest) -> None:
        """Update counters and hand the call to the usage sink."""
        self.stats.usage = self.stats.usage + completion.usage
        self.stats.cost_usd += completion.cost_usd
        if self._budget_remaining is not None:
            self._budget_remaining -= completion.usage.total_tokens
        if self._usage_sink is not None:
            await self._usage_sink(completion, request)


#: Failures that will fail identically no matter how many times they are tried.
#: Retrying these only burns the user's latency budget.
_PERMANENT_FAILURE_MARKERS: tuple[str, ...] = (
    "401",
    "403",
    "invalid api key",
    "invalid_api_key",
    "authentication",
    "permission",
    "invalid_request",
    "context_length",
    "model_not_found",
    "not configured",
)


def _is_retryable(error: ProviderError) -> bool:
    """Whether a provider failure is worth retrying.

    The default is *retry*: unrecognised failures are usually transport noise,
    and the cost of one extra attempt against a genuinely permanent error is a
    little latency, whereas the cost of not retrying a transient error is a
    failed user request. Only the explicitly enumerated permanent failures —
    bad credentials, malformed requests, unknown models — short-circuit.

    Example:
        >>> _is_retryable(ProviderError(provider="p", reason="429 rate_limit_error"))
        True
        >>> _is_retryable(ProviderError(provider="p", reason="connection reset"))
        True
        >>> _is_retryable(ProviderError(provider="p", reason="401 invalid api key"))
        False
    """
    reason = str(error.details.get("reason", "")).lower()
    return not any(marker in reason for marker in _PERMANENT_FAILURE_MARKERS)


def trim_to_context(
    messages: Sequence[Message], *, model: str, reserve_tokens: int
) -> list[Message]:
    """Drop the oldest non-system messages until the prompt fits the context.

    Uses a 4-characters-per-token approximation rather than a real tokeniser:
    this runs on every turn, exactness buys nothing (the reserve absorbs the
    error), and importing tiktoken here would couple the router to one vendor's
    tokenisation.

    System messages are never dropped — they carry the tenant's instructions and
    the guardrail framing.

    Example:
        >>> msgs = [Message.system("sys"), Message.user("a" * 400), Message.user("recent")]
        >>> kept = trim_to_context(msgs, model="gpt-4o", reserve_tokens=127_950)
        >>> [m.content for m in kept]
        ['sys', 'recent']
    """
    budget = context_window(model) - reserve_tokens
    if budget <= 0:
        return [m for m in messages if m.role.value == "system"]

    system = [m for m in messages if m.role.value == "system"]
    rest = [m for m in messages if m.role.value != "system"]

    used = sum(len(m.content) for m in system) // 4
    kept: list[Message] = []
    # Walk backwards so the most recent turns survive.
    for message in reversed(rest):
        cost = len(message.content) // 4 + 4
        if used + cost > budget:
            break
        used += cost
        kept.append(message)

    return system + list(reversed(kept))


def supports_tools(model: str) -> bool:
    """Whether tools may be attached to a request for this model.

    Example:
        >>> supports_tools("gpt-4o")
        True
    """
    return supports(model, "tools")
