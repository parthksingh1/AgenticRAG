"""Cache keys, statistics and the exact-match cache.

Cache keys are the whole design. A key that is too coarse serves a stale or
wrong-tenant answer; a key that is too fine never hits. :func:`build_cache_key`
includes everything that could change the answer — tenant, model, prompt
version, retrieval strategy and any filters — so a prompt edit or a strategy
change invalidates the cache automatically rather than requiring someone to
remember to flush it.

Tenant id is in every key. Serving one tenant's answer to another is the worst
bug this system could have, and a shared key namespace makes it a one-line
mistake away.

Example:
    >>> a = build_cache_key(query="hi", tenant_id="t1", model="m", prompt_version="v1")
    >>> b = build_cache_key(query="hi", tenant_id="t2", model="m", prompt_version="v1")
    >>> a == b
    False
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)

#: Cache entries older than this are ignored even if Redis has not expired them,
#: so a TTL change takes effect immediately rather than after the old TTL runs out.
DEFAULT_TTL_SECONDS = 3600


def build_cache_key(
    *,
    query: str,
    tenant_id: str,
    model: str,
    prompt_version: str | None = None,
    strategy: str | None = None,
    filters: dict[str, Any] | None = None,
    namespace: str = "answer",
) -> str:
    """Build a deterministic cache key from everything that affects the answer.

    Normalises the query's whitespace and case so "What is RAG?" and "what is
    rag?" share an entry, but nothing else is normalised — punctuation and word
    order can change meaning.

    Example:
        >>> k1 = build_cache_key(query="What is RAG?", tenant_id="t", model="m")
        >>> k2 = build_cache_key(query="what is rag?", tenant_id="t", model="m")
        >>> k1 == k2
        True
        >>> build_cache_key(query="q", tenant_id="t", model="m").startswith("answer:")
        True
    """
    payload = {
        "q": " ".join(query.lower().split()),
        "t": tenant_id,
        "m": model,
        "p": prompt_version or "",
        "s": strategy or "",
        "f": _stable(filters or {}),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{namespace}:{tenant_id}:{digest[:40]}"


def _stable(value: Any) -> Any:
    """Render a value in a form whose JSON encoding is order-independent.

    Example:
        >>> _stable({"b": [3, 1], "a": 1})
        {'a': 1, 'b': [1, 3]}
    """
    if isinstance(value, dict):
        return {k: _stable(value[k]) for k in sorted(value)}
    if isinstance(value, list | tuple | set):
        return sorted((_stable(v) for v in value), key=repr)
    return value


@dataclass(slots=True)
class CacheEntry:
    """A cached value with the metadata needed to trust it."""

    value: Any
    created_at: float
    #: What produced it, so a hit can be attributed in traces and cost reports.
    model: str | None = None
    prompt_version: str | None = None
    #: Present on semantic entries: the query text that was originally answered.
    source_query: str | None = None
    hit_count: int = 0

    def age_seconds(self, *, now: float | None = None) -> float:
        """How old the entry is.

        Example:
            >>> CacheEntry(value=1, created_at=100.0).age_seconds(now=160.0)
            60.0
        """
        return (time.time() if now is None else now) - self.created_at

    def is_fresh(self, ttl_seconds: int, *, now: float | None = None) -> bool:
        """Whether the entry is still within its time-to-live.

        Example:
            >>> CacheEntry(value=1, created_at=0.0).is_fresh(60, now=30.0)
            True
            >>> CacheEntry(value=1, created_at=0.0).is_fresh(60, now=90.0)
            False
        """
        return self.age_seconds(now=now) < ttl_seconds


@dataclass(slots=True)
class CacheStats:
    """Hit and miss counters for one cache.

    ``false_hits`` is the number the semantic cache lives or dies by: a cache
    that returns the wrong answer 5% of the time is far worse than no cache, and
    it is the only cache metric that cannot be inferred from hit rate alone.
    """

    hits: int = 0
    misses: int = 0
    false_hits: int = 0
    writes: int = 0
    errors: int = 0
    saved_cost_usd: float = 0.0
    similarities: list[float] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total lookups."""
        return self.hits + self.misses

    @property
    def hit_ratio(self) -> float:
        """Share of lookups served from cache.

        Example:
            >>> CacheStats(hits=3, misses=1).hit_ratio
            0.75
            >>> CacheStats().hit_ratio
            0.0
        """
        return self.hits / self.total if self.total else 0.0

    @property
    def false_hit_ratio(self) -> float:
        """Share of hits that were wrong and had to be rejected.

        Example:
            >>> CacheStats(hits=8, false_hits=2).false_hit_ratio
            0.2
        """
        served = self.hits + self.false_hits
        return self.false_hits / served if served else 0.0


class Cache(ABC):
    """Key-value cache with a TTL."""

    name: str

    @abstractmethod
    async def get(self, key: str) -> CacheEntry | None:
        """Return the entry, or None on a miss."""

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry, *, ttl_seconds: int) -> None:
        """Store an entry."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove an entry."""

    @abstractmethod
    async def clear_tenant(self, tenant_id: str) -> int:
        """Remove every entry for a tenant, returning the count.

        Required by GDPR erasure: a cached answer is still the tenant's data.
        """


class InMemoryCache(Cache):
    """Process-local cache for tests and single-process deployments."""

    name = "memory"

    def __init__(self) -> None:
        """Create an empty cache."""
        self._store: dict[str, tuple[CacheEntry, float]] = {}
        self.stats = CacheStats()

    async def get(self, key: str) -> CacheEntry | None:
        """Return a fresh entry, or None if missing or expired."""
        found = self._store.get(key)
        if found is None:
            self.stats.misses += 1
            return None
        entry, expires_at = found
        if time.time() >= expires_at:
            del self._store[key]
            self.stats.misses += 1
            return None
        entry.hit_count += 1
        self.stats.hits += 1
        return entry

    async def set(self, key: str, entry: CacheEntry, *, ttl_seconds: int) -> None:
        """Store an entry with an expiry."""
        self._store[key] = (entry, time.time() + ttl_seconds)
        self.stats.writes += 1

    async def delete(self, key: str) -> None:
        """Remove an entry if present."""
        self._store.pop(key, None)

    async def clear_tenant(self, tenant_id: str) -> int:
        """Drop every key belonging to a tenant."""
        prefix_marker = f":{tenant_id}:"
        doomed = [k for k in self._store if prefix_marker in k]
        for key in doomed:
            del self._store[key]
        return len(doomed)


class RedisCache(Cache):
    """Redis-backed cache shared across workers.

    A Redis failure is a miss, never an error. A cache exists to make things
    faster; letting it make things *fail* inverts the point, so every operation
    swallows its exception, records it in ``stats.errors`` and carries on.
    """

    name = "redis"

    def __init__(self, *, redis: Any, namespace: str = "cache") -> None:
        """Create the cache around an async Redis client."""
        self._redis = redis
        self._namespace = namespace
        self.stats = CacheStats()

    def _key(self, key: str) -> str:
        """Namespace a key."""
        return f"{self._namespace}:{key}"

    async def get(self, key: str) -> CacheEntry | None:
        """Read and decode an entry, treating any failure as a miss."""
        try:
            raw = await self._redis.get(self._key(key))
        except Exception as exc:  # noqa: BLE001 - a cache must not break the request
            self.stats.errors += 1
            log.warning("cache read failed; treating as a miss", reason=str(exc))
            return None

        if raw is None:
            self.stats.misses += 1
            return None

        try:
            payload = json.loads(raw)
            entry = CacheEntry(**payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            # A malformed entry is a bug or a schema change; drop it rather than
            # serving nonsense, and let the next request repopulate.
            self.stats.errors += 1
            log.warning("discarding malformed cache entry", key=key, reason=str(exc))
            await self.delete(key)
            return None

        self.stats.hits += 1
        return entry

    async def set(self, key: str, entry: CacheEntry, *, ttl_seconds: int) -> None:
        """Encode and store an entry."""
        try:
            payload = json.dumps(
                {
                    "value": entry.value,
                    "created_at": entry.created_at,
                    "model": entry.model,
                    "prompt_version": entry.prompt_version,
                    "source_query": entry.source_query,
                    "hit_count": entry.hit_count,
                }
            )
            await self._redis.set(self._key(key), payload, ex=ttl_seconds)
            self.stats.writes += 1
        except Exception as exc:  # noqa: BLE001 - a failed write is not a failed request
            self.stats.errors += 1
            log.warning("cache write failed", reason=str(exc))

    async def delete(self, key: str) -> None:
        """Remove an entry."""
        try:
            await self._redis.delete(self._key(key))
        except Exception as exc:  # noqa: BLE001 - best effort
            self.stats.errors += 1
            log.warning("cache delete failed", reason=str(exc))

    async def clear_tenant(self, tenant_id: str) -> int:
        """Scan and delete every key for a tenant.

        Uses SCAN rather than KEYS: KEYS blocks the Redis event loop for the
        duration, which on a large keyspace is an outage.
        """
        pattern = f"{self._namespace}:*:{tenant_id}:*"
        deleted = 0
        try:
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(cursor=cursor, match=pattern, count=500)
                if keys:
                    await self._redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
        except Exception as exc:  # noqa: BLE001 - erasure continues elsewhere
            self.stats.errors += 1
            log.error("tenant cache purge failed", tenant_id=tenant_id, reason=str(exc))
        else:
            log.warning("purged tenant cache", tenant_id=tenant_id, deleted=deleted)
        return deleted


class ToolResultCache:
    """Caches MCP tool results with a TTL chosen per tool.

    A calculator result is valid forever; a web fetch is stale in minutes; a SQL
    query over live data sits in between. One global TTL would either serve stale
    data or throw away results that could safely be reused, so the TTL is a
    property of the tool.
    """

    #: Tool name -> TTL seconds. Anything unlisted gets the conservative default.
    TTL_BY_TOOL: dict[str, int] = {  # noqa: RUF012 - configuration table
        "calculator": 86_400,
        "docs_search": 900,
        "sql_analytics": 300,
        "web_fetch": 600,
        "kg_query": 1800,
        "code_exec": 0,  # never cached: side effects and non-determinism
    }
    DEFAULT_TTL = 300

    def __init__(self, cache: Cache) -> None:
        """Wrap a cache backend."""
        self._cache = cache
        self.stats = CacheStats()

    def ttl_for(self, tool: str) -> int:
        """TTL for a tool, defaulting conservatively.

        Example:
            >>> ToolResultCache(InMemoryCache()).ttl_for("calculator")
            86400
            >>> ToolResultCache(InMemoryCache()).ttl_for("unknown_tool")
            300
        """
        return self.TTL_BY_TOOL.get(tool, self.DEFAULT_TTL)

    @staticmethod
    def key_for(*, tool: str, arguments: dict[str, Any], tenant_id: str) -> str:
        """Cache key for one tool invocation.

        Example:
            >>> k = ToolResultCache.key_for(tool="calc", arguments={"a": 1}, tenant_id="t")
            >>> k.startswith("tool:t:")
            True
        """
        digest = hashlib.sha256(
            json.dumps({"tool": tool, "args": _stable(arguments)}, sort_keys=True).encode()
        ).hexdigest()
        return f"tool:{tenant_id}:{digest[:40]}"

    async def get(self, *, tool: str, arguments: dict[str, Any], tenant_id: str) -> Any | None:
        """Return a cached tool result, or None."""
        if self.ttl_for(tool) <= 0:
            return None
        entry = await self._cache.get(
            self.key_for(tool=tool, arguments=arguments, tenant_id=tenant_id)
        )
        if entry is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return entry.value

    async def set(
        self, *, tool: str, arguments: dict[str, Any], tenant_id: str, result: Any
    ) -> None:
        """Store a tool result, unless the tool is marked uncacheable."""
        ttl = self.ttl_for(tool)
        if ttl <= 0:
            return
        await self._cache.set(
            self.key_for(tool=tool, arguments=arguments, tenant_id=tenant_id),
            CacheEntry(value=result, created_at=time.time()),
            ttl_seconds=ttl,
        )
        self.stats.writes += 1


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]; zero vectors are maximally dissimilar.

    Example:
        >>> cosine_similarity([1.0, 0.0], [1.0, 0.0])
        1.0
        >>> cosine_similarity([1.0, 0.0], [0.0, 1.0])
        0.0
        >>> cosine_similarity([0.0, 0.0], [1.0, 0.0])
        0.0
    """
    if len(a) != len(b):
        msg = f"vector length mismatch: {len(a)} != {len(b)}"
        raise ValueError(msg)
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
