"""Model pricing table and capability registry.

This is the single source of truth for what a call costs. Nothing in the system
is allowed to hard-code a price: cost dashboards, tenant budgets and eval reports
all resolve through :func:`price_for`, so a stale figure is one edit away from
being correct everywhere rather than scattered across the codebase.

Prices are USD per million tokens and were current at the time of writing; they
are deliberately data so they can be updated without touching logic. An unknown
model resolves to a conservative fallback and is logged, never silently priced
at zero — a zero-cost model would make budget enforcement fail open.

Example:
    >>> price_for("claude-sonnet-5").output_per_mtok > 0
    True
"""

from __future__ import annotations

from typing import Final

from src.core.logging import get_logger
from src.services.llm.types import ModelPricing

log = get_logger(__name__)


class ModelCapability:
    """Flags describing what a model can do, used by the router."""

    TOOLS = "tools"
    VISION = "vision"
    STREAMING = "streaming"
    JSON_SCHEMA = "json_schema"
    LONG_CONTEXT = "long_context"


#: model id -> provider name.
MODEL_PROVIDERS: Final[dict[str, str]] = {
    # Anthropic
    "claude-opus-5": "anthropic",
    "claude-sonnet-5": "anthropic",
    "claude-haiku-4-5-20251001": "anthropic",
    # OpenAI
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "text-embedding-3-large": "openai",
    # Google
    "gemini-2.0-flash": "google",
    # Groq / Together (OpenAI-compatible)
    "llama-3.3-70b-versatile": "groq",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "together",
}

#: model id -> pricing. USD per million tokens.
MODEL_PRICING: Final[dict[str, ModelPricing]] = {
    "claude-opus-5": ModelPricing(
        input_per_mtok=15.0, output_per_mtok=75.0, cached_input_per_mtok=1.5
    ),
    "claude-sonnet-5": ModelPricing(
        input_per_mtok=3.0, output_per_mtok=15.0, cached_input_per_mtok=0.3
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        input_per_mtok=1.0, output_per_mtok=5.0, cached_input_per_mtok=0.1
    ),
    "gpt-4o": ModelPricing(input_per_mtok=2.5, output_per_mtok=10.0, cached_input_per_mtok=1.25),
    "gpt-4o-mini": ModelPricing(
        input_per_mtok=0.15, output_per_mtok=0.6, cached_input_per_mtok=0.075
    ),
    "text-embedding-3-large": ModelPricing(input_per_mtok=0.13, output_per_mtok=0.0),
    "gemini-2.0-flash": ModelPricing(input_per_mtok=0.1, output_per_mtok=0.4),
    "llama-3.3-70b-versatile": ModelPricing(input_per_mtok=0.59, output_per_mtok=0.79),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": ModelPricing(
        input_per_mtok=0.88, output_per_mtok=0.88
    ),
}

#: Deliberately expensive, so an unpriced model shows up in the cost dashboard
#: as an anomaly rather than disappearing.
FALLBACK_PRICING: Final = ModelPricing(input_per_mtok=15.0, output_per_mtok=75.0)

#: model id -> capabilities.
MODEL_CAPABILITIES: Final[dict[str, frozenset[str]]] = {
    "claude-opus-5": frozenset(
        {
            ModelCapability.TOOLS,
            ModelCapability.VISION,
            ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT,
        }
    ),
    "claude-sonnet-5": frozenset(
        {
            ModelCapability.TOOLS,
            ModelCapability.VISION,
            ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT,
        }
    ),
    "claude-haiku-4-5-20251001": frozenset(
        {ModelCapability.TOOLS, ModelCapability.VISION, ModelCapability.STREAMING}
    ),
    "gpt-4o": frozenset(
        {
            ModelCapability.TOOLS,
            ModelCapability.VISION,
            ModelCapability.STREAMING,
            ModelCapability.JSON_SCHEMA,
        }
    ),
    "gpt-4o-mini": frozenset(
        {
            ModelCapability.TOOLS,
            ModelCapability.VISION,
            ModelCapability.STREAMING,
            ModelCapability.JSON_SCHEMA,
        }
    ),
    "gemini-2.0-flash": frozenset(
        {ModelCapability.TOOLS, ModelCapability.VISION, ModelCapability.STREAMING}
    ),
    "llama-3.3-70b-versatile": frozenset({ModelCapability.TOOLS, ModelCapability.STREAMING}),
}

#: Context windows, in tokens. Used to decide whether a prompt must be trimmed.
MODEL_CONTEXT_WINDOW: Final[dict[str, int]] = {
    "claude-opus-5": 200_000,
    "claude-sonnet-5": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gemini-2.0-flash": 1_000_000,
    "llama-3.3-70b-versatile": 128_000,
}

_warned_unknown: set[str] = set()


def price_for(model: str) -> ModelPricing:
    """Return pricing for a model, warning once for unknown ids.

    Example:
        >>> price_for("gpt-4o-mini").input_per_mtok
        0.15
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        if model not in _warned_unknown:
            _warned_unknown.add(model)
            log.warning(
                "no pricing entry for model; billing at the conservative fallback rate",
                model=model,
            )
        return FALLBACK_PRICING
    return pricing


def provider_for(model: str) -> str:
    """Return the provider that serves a model.

    Raises:
        KeyError: for an unregistered model, so a typo in tenant config fails
            at routing time rather than as a confusing provider 404.

    Example:
        >>> provider_for("claude-sonnet-5")
        'anthropic'
    """
    try:
        return MODEL_PROVIDERS[model]
    except KeyError:
        known = ", ".join(sorted(MODEL_PROVIDERS))
        msg = f"unknown model {model!r}; known models: {known}"
        raise KeyError(msg) from None


def supports(model: str, capability: str) -> bool:
    """Whether a model advertises a capability.

    Unknown models report no capabilities, so the router degrades to the simplest
    request shape rather than sending tools to a model that cannot use them.

    Example:
        >>> supports("gpt-4o", ModelCapability.JSON_SCHEMA)
        True
        >>> supports("mystery-model", ModelCapability.TOOLS)
        False
    """
    return capability in MODEL_CAPABILITIES.get(model, frozenset())


def context_window(model: str) -> int:
    """Context window in tokens, defaulting to a conservative 32k."""
    return MODEL_CONTEXT_WINDOW.get(model, 32_000)


def estimate_cost(model: str, *, prompt_tokens: int, completion_tokens: int) -> float:
    """Convenience wrapper for pricing a call from raw token counts.

    Example:
        >>> round(estimate_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500), 6)
        0.00045
    """
    from src.services.llm.types import Usage

    return price_for(model).cost_for(
        Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    )
