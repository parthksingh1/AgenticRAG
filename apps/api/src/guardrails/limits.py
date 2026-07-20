"""Rate limiting and cost caps.

A token bucket rather than a fixed window, because fixed windows allow a client
to send a full window's traffic in the last second of one window and again in
the first second of the next — double the intended rate at exactly the moment
the system is least able to absorb it. A bucket smooths that out and still
permits a legitimate burst up to its capacity.

The bucket lives in Redis and is refilled **lazily**: rather than a background
job topping up every tenant's bucket on a timer, the level is computed from the
elapsed time on each check. That means no scheduler, no drift, and correct
behaviour for a tenant that has been idle for a week.

Refill and consume happen in one Lua script so the read-modify-write is atomic.
Doing it in three round trips lets two concurrent requests both observe the same
level and both spend it — which under load is precisely when the limit matters.

Example:
    >>> bucket = InMemoryTokenBucket(capacity=2, refill_per_second=1)
    >>> import asyncio
    >>> asyncio.run(bucket.consume("t", 1)).allowed
    True
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.core.errors import BudgetExceededError, RateLimitedError
from src.core.logging import get_logger

log = get_logger(__name__)

#: Atomic lazy-refill token bucket.
#: KEYS[1] = bucket key
#: ARGV = capacity, refill_per_second, now (seconds, float), requested tokens, ttl
_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

local state = redis.call('HMGET', key, 'tokens', 'updated')
local tokens = tonumber(state[1])
local updated = tonumber(state[2])

if tokens == nil then
    tokens = capacity
    updated = now
end

local elapsed = math.max(now - updated, 0)
tokens = math.min(capacity, tokens + elapsed * refill)

local allowed = 0
local retry_after = 0
if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
else
    retry_after = (requested - tokens) / refill
end

redis.call('HMSET', key, 'tokens', tokens, 'updated', now)
redis.call('EXPIRE', key, ttl)
return {allowed, tostring(tokens), tostring(retry_after)}
"""


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """The outcome of one rate-limit check."""

    allowed: bool
    remaining: float
    retry_after_seconds: float

    def raise_if_limited(self) -> None:
        """Raise the standard 429 error when the request is not allowed.

        Raises:
            RateLimitedError: carrying the Retry-After hint.
        """
        if not self.allowed:
            raise RateLimitedError(retry_after_seconds=self.retry_after_seconds)


class TokenBucket:
    """Redis-backed token bucket, one per tenant per endpoint."""

    def __init__(
        self,
        *,
        redis: Any,
        capacity: int,
        refill_per_second: float,
        namespace: str = "rl",
    ) -> None:
        """Create a bucket.

        Args:
            redis: An async Redis client.
            capacity: Burst size — the most that can be spent at once.
            refill_per_second: Sustained rate.
            namespace: Key prefix, so endpoints get independent buckets.
        """
        if capacity < 1:
            msg = "capacity must be at least 1"
            raise ValueError(msg)
        if refill_per_second <= 0:
            msg = "refill_per_second must be positive"
            raise ValueError(msg)

        self._redis = redis
        self._capacity = capacity
        self._refill = refill_per_second
        self._namespace = namespace
        self._script: Any | None = None

    async def consume(self, key: str, tokens: float = 1.0) -> RateLimitDecision:
        """Attempt to spend tokens, refilling lazily first.

        A Redis outage allows the request rather than denying it: rate limiting
        protects capacity, and failing closed would convert a cache outage into a
        total outage. The event is logged so the gap is visible.
        """
        if self._script is None:
            self._script = self._redis.register_script(_BUCKET_LUA)

        # TTL long enough that a bucket survives a quiet period but is eventually
        # reclaimed for tenants that stop sending traffic.
        ttl = int(self._capacity / self._refill) + 60
        try:
            allowed, remaining, retry_after = await self._script(
                keys=[f"{self._namespace}:{key}"],
                args=[self._capacity, self._refill, time.time(), tokens, ttl],
            )
        except Exception as exc:  # noqa: BLE001 - never let Redis take down the API
            log.error("rate limiter unavailable; allowing request", reason=str(exc))
            return RateLimitDecision(
                allowed=True, remaining=float(self._capacity), retry_after_seconds=0.0
            )

        return RateLimitDecision(
            allowed=bool(int(allowed)),
            remaining=float(remaining),
            retry_after_seconds=float(retry_after),
        )


class InMemoryTokenBucket:
    """Process-local token bucket for tests and single-process deployments.

    Same lazy-refill semantics as the Redis version, so the tests exercise real
    behaviour rather than a simplification.
    """

    def __init__(self, *, capacity: int, refill_per_second: float) -> None:
        """Create a bucket."""
        if capacity < 1:
            msg = "capacity must be at least 1"
            raise ValueError(msg)
        if refill_per_second <= 0:
            msg = "refill_per_second must be positive"
            raise ValueError(msg)
        self._capacity = float(capacity)
        self._refill = refill_per_second
        self._state: dict[str, tuple[float, float]] = {}

    async def consume(
        self, key: str, tokens: float = 1.0, *, now: float | None = None
    ) -> RateLimitDecision:
        """Attempt to spend tokens.

        Args:
            key: Bucket identity, usually ``tenant_id:endpoint``.
            tokens: How many to spend.
            now: Injectable clock, so tests can advance time without sleeping.

        Example:
            >>> import asyncio
            >>> bucket = InMemoryTokenBucket(capacity=1, refill_per_second=1)
            >>> asyncio.run(bucket.consume("k", now=0.0)).allowed
            True
            >>> asyncio.run(bucket.consume("k", now=0.0)).allowed
            False
            >>> asyncio.run(bucket.consume("k", now=1.0)).allowed
            True
        """
        current = time.time() if now is None else now
        level, updated = self._state.get(key, (self._capacity, current))
        level = min(self._capacity, level + max(current - updated, 0.0) * self._refill)

        if level >= tokens:
            self._state[key] = (level - tokens, current)
            return RateLimitDecision(
                allowed=True, remaining=level - tokens, retry_after_seconds=0.0
            )

        self._state[key] = (level, current)
        return RateLimitDecision(
            allowed=False,
            remaining=level,
            retry_after_seconds=(tokens - level) / self._refill,
        )


@dataclass(slots=True)
class BudgetStatus:
    """A tenant's spend against its daily budget."""

    tokens_used: int
    tokens_limit: int
    cost_usd: float
    cost_limit_usd: float | None = None

    @property
    def tokens_remaining(self) -> int:
        """Tokens left today, never negative."""
        return max(self.tokens_limit - self.tokens_used, 0)

    @property
    def exhausted(self) -> bool:
        """Whether either the token or the cost ceiling has been reached.

        Example:
            >>> BudgetStatus(tokens_used=100, tokens_limit=100, cost_usd=0).exhausted
            True
            >>> BudgetStatus(tokens_used=50, tokens_limit=100, cost_usd=0).exhausted
            False
        """
        if self.tokens_used >= self.tokens_limit:
            return True
        return self.cost_limit_usd is not None and self.cost_usd >= self.cost_limit_usd

    @property
    def fraction_used(self) -> float:
        """Share of the token budget consumed, clamped to 1.0."""
        if self.tokens_limit <= 0:
            return 1.0
        return min(self.tokens_used / self.tokens_limit, 1.0)

    def raise_if_exhausted(self) -> None:
        """Raise when the budget is spent.

        Raises:
            BudgetExceededError: identifying which ceiling was hit.
        """
        if self.cost_limit_usd is not None and self.cost_usd >= self.cost_limit_usd:
            raise BudgetExceededError(
                limit=self.cost_limit_usd, used=self.cost_usd, window="monthly_cost"
            )
        if self.tokens_used >= self.tokens_limit:
            raise BudgetExceededError(
                limit=self.tokens_limit, used=self.tokens_used, window="daily_tokens"
            )


class BudgetTracker:
    """Tracks per-tenant daily token spend in Redis.

    Redis holds the hot counter; :class:`~src.models.telemetry.TenantBudgetCounter`
    is the durable record. The split matters because a Redis flush must not hand
    every tenant an unlimited budget, and a database write on every token would
    not survive production traffic.
    """

    def __init__(self, *, redis: Any, namespace: str = "budget") -> None:
        """Create the tracker."""
        self._redis = redis
        self._namespace = namespace

    def _key(self, tenant_id: str, day: str) -> str:
        """Redis key for one tenant-day."""
        return f"{self._namespace}:{tenant_id}:{day}"

    async def record(self, tenant_id: str, *, tokens: int, cost_usd: float, day: str) -> None:
        """Add spend to today's counter."""
        key = self._key(tenant_id, day)
        try:
            pipe = self._redis.pipeline()
            pipe.hincrby(key, "tokens", tokens)
            pipe.hincrbyfloat(key, "cost", cost_usd)
            # Two days, so a request that straddles midnight still resolves.
            pipe.expire(key, 172_800)
            await pipe.execute()
        except Exception as exc:  # noqa: BLE001 - accounting must not fail a turn
            log.error("failed to record budget usage", tenant_id=tenant_id, reason=str(exc))

    async def status(
        self,
        tenant_id: str,
        *,
        day: str,
        tokens_limit: int,
        cost_limit_usd: float | None = None,
    ) -> BudgetStatus:
        """Read a tenant's spend for a day.

        On a Redis failure this reports zero usage rather than blocking the
        tenant: refusing paid traffic because a cache is down is the wrong
        trade, and the durable counter still records the truth.
        """
        try:
            raw = await self._redis.hgetall(self._key(tenant_id, day))
        except Exception as exc:  # noqa: BLE001 - never block on a cache outage
            log.error(
                "budget lookup failed; assuming unspent", tenant_id=tenant_id, reason=str(exc)
            )
            raw = {}

        return BudgetStatus(
            tokens_used=int(_field(raw, "tokens", 0)),
            tokens_limit=tokens_limit,
            cost_usd=float(_field(raw, "cost", 0.0)),
            cost_limit_usd=cost_limit_usd,
        )


def _field(mapping: dict[Any, Any], name: str, default: float) -> float:
    """Read a Redis hash field, tolerating bytes keys and missing values.

    Example:
        >>> _field({b"tokens": b"42"}, "tokens", 0)
        42.0
        >>> _field({}, "tokens", 0)
        0.0
    """
    for key, value in mapping.items():
        key_text = key.decode() if isinstance(key, bytes) else str(key)
        if key_text == name:
            raw = value.decode() if isinstance(value, bytes) else value
            try:
                return float(raw)
            except (TypeError, ValueError):
                return float(default)
    return float(default)
