"""LLM router tests: retry, fallback, budgets and cost-aware routing.

Every provider call spends money, so the behaviours pinned here are the ones
that decide whether a bad afternoon costs a few retries or a few thousand
dollars.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from src.core.errors import BudgetExceededError, ProviderError
from src.services.llm import pricing
from src.services.llm.providers import FakeProvider
from src.services.llm.router import (
    LLMRouter,
    ModelPolicy,
    RetryConfig,
    _is_retryable,
    trim_to_context,
)
from src.services.llm.types import (
    Completion,
    CompletionRequest,
    Message,
    ModelPricing,
    StreamEventType,
    Usage,
)

pytestmark = pytest.mark.unit

PRIMARY = "test-primary"
BACKUP = "test-backup"
CHEAP = "test-cheap"


@pytest.fixture(autouse=True)
def _register_test_models() -> Iterator[None]:
    """Register throwaway models so the router can resolve their providers."""
    added = {PRIMARY: "alpha", BACKUP: "beta", CHEAP: "alpha"}
    pricing.MODEL_PROVIDERS.update(added)
    for model in added:
        pricing.MODEL_PRICING[model] = ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0)
    yield
    for model in added:
        pricing.MODEL_PROVIDERS.pop(model, None)
        pricing.MODEL_PRICING.pop(model, None)


def request(text: str = "question") -> CompletionRequest:
    """A minimal completion request against the primary model."""
    return CompletionRequest(messages=(Message.user(text),), model=PRIMARY)


def router(
    providers: dict[str, FakeProvider],
    *,
    policy: ModelPolicy | None = None,
    attempts: int = 4,
    budget: int | None = None,
) -> LLMRouter:
    """A router wired to fake providers with near-zero backoff."""
    return LLMRouter(
        providers=dict(providers),
        policy=policy or ModelPolicy(default_model=PRIMARY),
        retry=RetryConfig(max_attempts=attempts, initial_delay_seconds=0.001),
        budget_remaining_tokens=budget,
    )


# ── Retry ────────────────────────────────────────────────────────────────────


async def test_transient_failures_are_retried_then_succeed() -> None:
    """Two blips followed by a success must produce an answer, not an error."""
    provider = FakeProvider(responses=["recovered"], fail_times=2)
    r = router({"alpha": provider})

    completion = await r.complete(request())

    assert completion.content == "recovered"
    assert r.stats.retries == 2
    assert r.stats.calls == 3


async def test_retries_are_bounded_by_max_attempts() -> None:
    """A permanently unhealthy provider fails after exactly max_attempts tries."""
    provider = FakeProvider(fail_times=99)
    r = router({"alpha": provider}, attempts=3)

    with pytest.raises(ProviderError):
        await r.complete(request(), allow_fallback=False)

    assert len(provider.calls) == 3


async def test_permanent_failures_are_not_retried() -> None:
    """Bad credentials fail identically forever; retrying only adds latency."""

    async def always_unauthorised(_request: CompletionRequest) -> Completion:
        raise ProviderError(provider="alpha", reason="401 invalid api key")

    provider = FakeProvider()
    provider.complete = always_unauthorised  # type: ignore[method-assign]
    r = router({"alpha": provider}, attempts=5)

    with pytest.raises(ProviderError):
        await r.complete(request(), allow_fallback=False)

    assert r.stats.retries == 0


@pytest.mark.parametrize(
    ("reason", "retryable"),
    [
        ("429 rate_limit_error", True),
        ("503 overloaded", True),
        ("connection reset by peer", True),
        ("read timeout", True),
        ("something nobody has seen before", True),
        ("401 invalid api key", False),
        ("403 permission denied", False),
        ("invalid_request: bad tool schema", False),
        ("context_length exceeded", False),
    ],
)
def test_retry_classification(reason: str, retryable: bool) -> None:
    """Unknown failures retry by default; enumerated permanent ones do not."""
    assert _is_retryable(ProviderError(provider="p", reason=reason)) is retryable


def test_backoff_grows_then_saturates() -> None:
    """Delay grows geometrically and is clamped, so a long outage cannot stall forever."""
    config = RetryConfig(initial_delay_seconds=0.5, multiplier=2.0, max_delay_seconds=8.0)

    assert config.delay_for(1, jitter=1.0) == 0.5
    assert config.delay_for(3, jitter=1.0) == 2.0
    assert config.delay_for(99, jitter=1.0) == 8.0


def test_backoff_applies_jitter() -> None:
    """Full jitter is what stops every worker retrying on the same tick."""
    config = RetryConfig(initial_delay_seconds=1.0)
    assert config.delay_for(1, jitter=0.25) == 0.25


# ── Fallback ─────────────────────────────────────────────────────────────────


async def test_falls_back_to_a_second_provider() -> None:
    """When the primary is down, a different provider answers the same request."""
    dead = FakeProvider(fail_times=99)
    alive = FakeProvider(responses=["from backup"])
    r = router(
        {"alpha": dead, "beta": alive},
        policy=ModelPolicy(default_model=PRIMARY, fallback_models=(BACKUP,)),
        attempts=2,
    )

    completion = await r.complete(request())

    assert completion.content == "from backup"
    assert completion.was_fallback is True
    assert r.stats.fallbacks == 1


async def test_fallback_response_is_marked_so_evals_can_tell() -> None:
    """An unmarked fallback would silently compare two different models."""
    dead = FakeProvider(fail_times=99)
    alive = FakeProvider(responses=["backup"])
    r = router(
        {"alpha": dead, "beta": alive},
        policy=ModelPolicy(default_model=PRIMARY, fallback_models=(BACKUP,)),
        attempts=1,
    )

    assert (await r.complete(request())).was_fallback is True


async def test_a_healthy_primary_is_not_marked_as_fallback() -> None:
    """The flag must mean something, so the happy path never sets it."""
    r = router(
        {"alpha": FakeProvider(responses=["ok"]), "beta": FakeProvider()},
        policy=ModelPolicy(default_model=PRIMARY, fallback_models=(BACKUP,)),
    )

    assert (await r.complete(request())).was_fallback is False


async def test_fallback_can_be_disabled_per_call() -> None:
    """Internal calls (the complexity classifier) must not escalate to a pricier model."""
    dead = FakeProvider(fail_times=99)
    alive = FakeProvider(responses=["backup"])
    r = router(
        {"alpha": dead, "beta": alive},
        policy=ModelPolicy(default_model=PRIMARY, fallback_models=(BACKUP,)),
        attempts=1,
    )

    with pytest.raises(ProviderError):
        await r.complete(request(), allow_fallback=False)

    assert alive.calls == []


async def test_exhausting_every_model_raises_with_the_chain_in_the_message() -> None:
    """The operator needs to know what was tried, not just that it failed."""
    r = router(
        {"alpha": FakeProvider(fail_times=99), "beta": FakeProvider(fail_times=99)},
        policy=ModelPolicy(default_model=PRIMARY, fallback_models=(BACKUP,)),
        attempts=1,
    )

    with pytest.raises(ProviderError, match=PRIMARY):
        await r.complete(request())


async def test_missing_provider_is_reported_as_configuration_not_as_a_blip() -> None:
    """A missing API key must not be retried four times before surfacing."""
    r = router({}, attempts=3)

    with pytest.raises(ProviderError, match="not configured"):
        await r.complete(request(), allow_fallback=False)


# ── Policy ───────────────────────────────────────────────────────────────────


def test_policy_allowlist_blocks_unlisted_models() -> None:
    """A tenant restricted to one model cannot be steered onto another."""
    policy = ModelPolicy(default_model=PRIMARY, allowed_models=(PRIMARY,))

    assert policy.allows(PRIMARY) is True
    assert policy.allows(BACKUP) is False


def test_policy_resolves_a_disallowed_model_to_the_default() -> None:
    """Tightening an allowlist mid-conversation degrades rather than breaks."""
    policy = ModelPolicy(default_model=PRIMARY, allowed_models=(PRIMARY,))

    assert policy.resolve(BACKUP) == PRIMARY


def test_empty_allowlist_permits_everything() -> None:
    """An unset allowlist is "no restriction", not "nothing allowed"."""
    assert ModelPolicy(default_model=PRIMARY).allows("any-model") is True


async def test_fallback_chain_skips_models_the_tenant_may_not_use() -> None:
    """Fallback must respect the allowlist, not bypass it under pressure."""
    r = router(
        {"alpha": FakeProvider(fail_times=99), "beta": FakeProvider(responses=["backup"])},
        policy=ModelPolicy(
            default_model=PRIMARY, fallback_models=(BACKUP,), allowed_models=(PRIMARY,)
        ),
        attempts=1,
    )

    with pytest.raises(ProviderError):
        await r.complete(request())


# ── Cost-aware routing ───────────────────────────────────────────────────────


async def test_simple_queries_are_downgraded_to_the_cheap_model() -> None:
    """The largest cost lever in the system: most questions are not hard."""
    r = router(
        {"alpha": FakeProvider(responses=["SIMPLE"])},
        policy=ModelPolicy(default_model=PRIMARY, cheap_model=CHEAP),
    )

    assert await r.select_model("what is the capital of France?") == CHEAP
    assert r.stats.downgrades == 1


async def test_complex_queries_keep_the_frontier_model() -> None:
    """A hard question must not be quietly served by the mini model."""
    r = router(
        {"alpha": FakeProvider(responses=["COMPLEX"])},
        policy=ModelPolicy(default_model=PRIMARY, cheap_model=CHEAP),
    )

    assert await r.select_model("compare these three architectures and justify a choice") == PRIMARY


async def test_classifier_failure_defaults_to_the_frontier_model() -> None:
    """Under-serving a hard question is worse than over-paying for an easy one."""
    r = router(
        {"alpha": FakeProvider(fail_times=99)},
        policy=ModelPolicy(default_model=PRIMARY, cheap_model=CHEAP),
        attempts=1,
    )

    assert await r.classify_complexity("anything") is True


async def test_cost_routing_can_be_switched_off() -> None:
    """Tenants who want deterministic model choice get it without code changes."""
    provider = FakeProvider(responses=["SIMPLE"])
    r = router(
        {"alpha": provider},
        policy=ModelPolicy(default_model=PRIMARY, cheap_model=CHEAP, cost_aware_routing=False),
    )

    assert await r.select_model("trivial") == PRIMARY
    assert provider.calls == [], "classifier must not be called when routing is off"


async def test_an_explicitly_requested_model_skips_the_classifier() -> None:
    """A user picking a model in the UI must not be second-guessed."""
    provider = FakeProvider(responses=["SIMPLE"])
    r = router({"alpha": provider}, policy=ModelPolicy(default_model=PRIMARY, cheap_model=CHEAP))

    assert await r.select_model("trivial", requested=PRIMARY) == PRIMARY
    assert provider.calls == []


# ── Budgets and accounting ───────────────────────────────────────────────────


async def test_every_call_is_costed_and_accumulated() -> None:
    """No path may spend money without attribution."""
    r = router({"alpha": FakeProvider(responses=["hello"])})

    completion = await r.complete(request())

    assert completion.cost_usd > 0
    assert r.stats.cost_usd == pytest.approx(completion.cost_usd)
    assert r.stats.usage.total_tokens == completion.usage.total_tokens


async def test_usage_sink_receives_every_completion() -> None:
    """The service layer persists a UsageRecord without the router touching the DB."""
    seen: list[Completion] = []

    async def sink(completion: Completion, _request: CompletionRequest) -> None:
        seen.append(completion)

    r = LLMRouter(
        providers={"alpha": FakeProvider(responses=["a", "b"])},
        policy=ModelPolicy(default_model=PRIMARY),
        usage_sink=sink,
    )
    await r.complete(request())
    await r.complete(request())

    assert len(seen) == 2


async def test_exhausted_budget_fails_before_spending() -> None:
    """The check happens before the provider call, not after the bill arrives."""
    provider = FakeProvider(responses=["expensive"])
    r = router({"alpha": provider}, budget=0)

    with pytest.raises(BudgetExceededError):
        await r.complete(request())

    assert provider.calls == []


async def test_oversized_request_is_rejected_against_the_per_request_cap() -> None:
    """A single runaway request cannot consume a whole tenant's daily budget."""
    r = LLMRouter(
        providers={"alpha": FakeProvider()},
        policy=ModelPolicy(default_model=PRIMARY, max_tokens_per_request=100),
        budget_remaining_tokens=1_000_000,
    )
    oversized = CompletionRequest(messages=(Message.user("q"),), model=PRIMARY, max_tokens=5000)

    with pytest.raises(BudgetExceededError):
        await r.complete(oversized)


async def test_budget_is_decremented_as_calls_are_made() -> None:
    """A long agent turn eventually runs out rather than looping forever."""
    r = router({"alpha": FakeProvider(responses=["x" * 400])}, budget=50)

    await r.complete(request())

    with pytest.raises(BudgetExceededError):
        await r.complete(request())


# ── Streaming ────────────────────────────────────────────────────────────────


async def test_stream_emits_text_then_usage_then_done() -> None:
    """The frontend depends on this event order to close out a message."""
    r = router({"alpha": FakeProvider(responses=["one two three"])})

    events = [event async for event in r.stream(request())]

    assert events[-1].type is StreamEventType.DONE
    assert any(e.type is StreamEventType.USAGE for e in events)
    text = "".join(e.text for e in events if e.type is StreamEventType.TEXT)
    assert text.strip() == "one two three"


async def test_stream_failing_before_the_first_token_is_retried_once() -> None:
    """A connection reset at open is recoverable; the user sees no error."""
    attempts = {"n": 0}

    class FlakyStream(FakeProvider):
        async def stream(self, req: CompletionRequest) -> AsyncIterator[Any]:  # type: ignore[override]
            attempts["n"] += 1
            if attempts["n"] == 1:
                from src.services.llm.types import StreamEvent

                yield StreamEvent(type=StreamEventType.ERROR, error="connection reset")
                return
            async for event in super().stream(req):
                yield event

    r = router({"alpha": FlakyStream(responses=["recovered"])})
    events = [event async for event in r.stream(request())]

    assert attempts["n"] == 2
    assert not any(e.type is StreamEventType.ERROR for e in events)


async def test_stream_failing_mid_answer_is_not_silently_restarted() -> None:
    """Splicing two different answers together is worse than showing the error."""

    class BreaksMidway(FakeProvider):
        async def stream(self, req: CompletionRequest) -> AsyncIterator[Any]:  # type: ignore[override]
            from src.services.llm.types import StreamEvent

            yield StreamEvent(type=StreamEventType.TEXT, text="partial ")
            yield StreamEvent(type=StreamEventType.ERROR, error="connection reset")

    r = router({"alpha": BreaksMidway()})
    events = [event async for event in r.stream(request())]

    assert events[-1].type is StreamEventType.ERROR


# ── Context trimming ─────────────────────────────────────────────────────────


def test_trimming_drops_oldest_turns_and_keeps_system_messages() -> None:
    """Tenant instructions and guardrail framing must survive any trim."""
    messages = [
        Message.system("tenant instructions"),
        Message.user("ancient history " * 100),
        Message.user("most recent question"),
    ]

    kept = trim_to_context(messages, model="gpt-4o", reserve_tokens=127_950)

    assert [m.content for m in kept] == ["tenant instructions", "most recent question"]


def test_trimming_a_fitting_conversation_changes_nothing() -> None:
    """The common case must be a no-op, not a silent reordering."""
    messages = [Message.system("s"), Message.user("a"), Message.assistant("b")]

    assert trim_to_context(messages, model="gpt-4o", reserve_tokens=1000) == messages


def test_trimming_with_no_budget_keeps_only_system_messages() -> None:
    """A pathological reserve degrades predictably instead of returning nothing."""
    messages = [Message.system("s"), Message.user("a")]

    kept = trim_to_context(messages, model="gpt-4o", reserve_tokens=999_999)

    assert [m.role.value for m in kept] == ["system"]


# ── Usage arithmetic ─────────────────────────────────────────────────────────


def test_usage_addition_is_component_wise() -> None:
    """Aggregating a multi-call turn must not lose the cached-token split."""
    total = Usage(prompt_tokens=10, completion_tokens=5, cached_tokens=2) + Usage(
        prompt_tokens=1, completion_tokens=1, cached_tokens=1
    )

    assert (total.prompt_tokens, total.completion_tokens, total.cached_tokens) == (11, 6, 3)


def test_cached_tokens_are_billed_at_the_cached_rate() -> None:
    """Prompt caching is only worth having if the cost model reflects it."""
    p = ModelPricing(input_per_mtok=10.0, output_per_mtok=0.0, cached_input_per_mtok=1.0)

    full = p.cost_for(Usage(prompt_tokens=1_000_000))
    cached = p.cost_for(Usage(prompt_tokens=1_000_000, cached_tokens=1_000_000))

    assert full == pytest.approx(10.0)
    assert cached == pytest.approx(1.0)


def test_unknown_models_price_conservatively_rather_than_free() -> None:
    """A zero-cost unknown model would make budget enforcement fail open."""
    assert pricing.price_for("not-a-real-model").input_per_mtok > 0
