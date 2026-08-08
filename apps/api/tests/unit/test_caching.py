"""Cache tests.

The semantic cache is the one that can be actively harmful: a near neighbour in
embedding space is not the same question, and serving the opposite answer is
worse than any latency saving. Most of these tests exist to pin that.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from src.caching.base import (
    CacheEntry,
    CacheStats,
    InMemoryCache,
    RedisCache,
    ToolResultCache,
    build_cache_key,
    cosine_similarity,
)
from src.caching.semantic import SemanticCache, differs_by_negation, normalise
from src.ingestion.embedders.base import HashingEmbedder
from src.services.llm import pricing
from src.services.llm.providers import FakeProvider
from src.services.llm.router import LLMRouter, ModelPolicy

pytestmark = pytest.mark.unit


class StaticEmbedder(HashingEmbedder):
    """Embeds by lookup, so tests can control similarity exactly."""

    def __init__(self, vectors: dict[str, list[float]], *, dimension: int = 3) -> None:
        """Create an embedder backed by a fixed lookup table."""
        super().__init__(dimension=dimension, model_name="static-test")
        self._vectors = vectors

    async def embed_query(self, text: str) -> list[float]:
        """Return the configured vector, or a zero vector."""
        return self._vectors.get(text, [0.0] * self.dimension)


def verify_router(answer: str) -> LLMRouter:
    """A router whose fake provider returns a fixed verification verdict."""
    pricing.MODEL_PROVIDERS["cache-verify-model"] = "fake"
    return LLMRouter(
        providers={"fake": FakeProvider(responses=[answer])},
        policy=ModelPolicy(default_model="cache-verify-model"),
    )


# ── Keys ─────────────────────────────────────────────────────────────────────


def test_tenants_never_share_a_cache_key() -> None:
    """Serving one tenant's answer to another is the worst bug this system could have."""
    a = build_cache_key(query="q", tenant_id="ten_a", model="m")
    b = build_cache_key(query="q", tenant_id="ten_b", model="m")

    assert a != b


def test_key_normalises_case_and_whitespace() -> None:
    """The same question typed differently must share an entry."""
    assert build_cache_key(query="What is RAG?", tenant_id="t", model="m") == build_cache_key(
        query="what is  rag?", tenant_id="t", model="m"
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"model": "different-model"},
        {"prompt_version": "v2"},
        {"strategy": "hybrid+rerank"},
        {"filters": {"document_ids": ["doc_1"]}},
    ],
)
def test_anything_that_changes_the_answer_changes_the_key(changed: dict[str, Any]) -> None:
    """A prompt edit must invalidate the cache without anyone remembering to flush it."""
    base: dict[str, Any] = {"query": "q", "tenant_id": "t", "model": "m", "prompt_version": "v1"}

    assert build_cache_key(**base) != build_cache_key(**{**base, **changed})


def test_filter_ordering_does_not_change_the_key() -> None:
    """Otherwise the same request misses depending on dict iteration order."""
    a = build_cache_key(query="q", tenant_id="t", model="m", filters={"a": [1, 2], "b": 1})
    b = build_cache_key(query="q", tenant_id="t", model="m", filters={"b": 1, "a": [2, 1]})

    assert a == b


# ── Entries and stats ────────────────────────────────────────────────────────


def test_entry_freshness_is_computed_from_its_age() -> None:
    """A TTL change must take effect immediately, not after the old TTL expires."""
    entry = CacheEntry(value=1, created_at=0.0)

    assert entry.is_fresh(60, now=30.0) is True
    assert entry.is_fresh(60, now=90.0) is False


def test_hit_ratio_of_an_unused_cache_is_zero_not_undefined() -> None:
    """The dashboard must not divide by zero on a fresh deployment."""
    assert CacheStats().hit_ratio == 0.0
    assert CacheStats().false_hit_ratio == 0.0


def test_false_hit_ratio_is_tracked_separately_from_hit_ratio() -> None:
    """A cache that is wrong 5% of the time is worse than no cache at all."""
    assert CacheStats(hits=8, false_hits=2).false_hit_ratio == pytest.approx(0.2)


# ── In-memory cache ──────────────────────────────────────────────────────────


async def test_stored_values_are_returned() -> None:
    """The basic contract."""
    cache = InMemoryCache()
    await cache.set("k", CacheEntry(value="answer", created_at=time.time()), ttl_seconds=60)

    entry = await cache.get("k")

    assert entry is not None
    assert entry.value == "answer"


async def test_expired_entries_are_treated_as_misses() -> None:
    """A stale answer served confidently is worse than a slow correct one."""
    cache = InMemoryCache()
    await cache.set("k", CacheEntry(value="v", created_at=time.time()), ttl_seconds=0)

    assert await cache.get("k") is None


async def test_missing_keys_count_as_misses() -> None:
    """Hit rate is only meaningful if misses are counted."""
    cache = InMemoryCache()

    await cache.get("absent")

    assert cache.stats.misses == 1


async def test_tenant_purge_removes_only_that_tenant() -> None:
    """GDPR erasure must not take another tenant's cache with it."""
    cache = InMemoryCache()
    entry = CacheEntry(value="v", created_at=time.time())
    await cache.set(build_cache_key(query="q", tenant_id="ten_a", model="m"), entry, ttl_seconds=60)
    await cache.set(build_cache_key(query="q", tenant_id="ten_b", model="m"), entry, ttl_seconds=60)

    removed = await cache.clear_tenant("ten_a")

    assert removed == 1
    assert await cache.get(build_cache_key(query="q", tenant_id="ten_b", model="m")) is not None


# ── Redis cache resilience ───────────────────────────────────────────────────


class FakeRedis:
    """Enough Redis for the cache, with a switchable failure mode."""

    def __init__(self, *, fail: bool = False) -> None:
        """Create the fake, optionally in permanent-failure mode."""
        self.fail = fail
        self.store: dict[str, str] = {}

    def _guard(self) -> None:
        """Raise when the fake is in failure mode."""
        if self.fail:
            msg = "redis down"
            raise RuntimeError(msg)

    async def get(self, key: str) -> str | None:
        """Read a key."""
        self._guard()
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        """Write a key."""
        self._guard()
        self.store[key] = value

    async def delete(self, *keys: str) -> None:
        """Delete keys."""
        self._guard()
        for key in keys:
            self.store.pop(key, None)

    async def scan(
        self, cursor: int = 0, match: str = "*", count: int = 100
    ) -> tuple[int, list[str]]:
        """Return every matching key in one pass."""
        self._guard()
        import fnmatch

        return 0, [k for k in list(self.store) if fnmatch.fnmatch(k, match)]


async def test_redis_cache_round_trips_an_entry() -> None:
    """The encode/decode path must preserve the value and its provenance."""
    cache = RedisCache(redis=FakeRedis())
    await cache.set("k", CacheEntry(value={"a": 1}, created_at=1.0, model="m"), ttl_seconds=60)

    entry = await cache.get("k")

    assert entry is not None
    assert entry.value == {"a": 1}
    assert entry.model == "m"


async def test_a_redis_outage_is_a_miss_not_an_error() -> None:
    """A cache that can make requests fail has inverted its own purpose."""
    cache = RedisCache(redis=FakeRedis(fail=True))

    assert await cache.get("k") is None
    assert cache.stats.errors == 1


async def test_a_failed_cache_write_does_not_raise() -> None:
    """The answer was already produced; failing to store it must not lose it."""
    cache = RedisCache(redis=FakeRedis(fail=True))

    await cache.set("k", CacheEntry(value="v", created_at=1.0), ttl_seconds=60)

    assert cache.stats.errors == 1


async def test_a_malformed_entry_is_discarded_rather_than_served() -> None:
    """A schema change must not make the cache serve nonsense."""
    redis = FakeRedis()
    cache = RedisCache(redis=redis)
    redis.store["cache:k"] = "{not valid json"

    assert await cache.get("k") is None
    assert "cache:k" not in redis.store


async def test_redis_tenant_purge_uses_scan() -> None:
    """KEYS blocks the Redis event loop, which on a large keyspace is an outage."""
    redis = FakeRedis()
    cache = RedisCache(redis=redis)
    entry = CacheEntry(value="v", created_at=time.time())
    await cache.set(build_cache_key(query="q", tenant_id="ten_a", model="m"), entry, ttl_seconds=60)
    await cache.set(build_cache_key(query="q", tenant_id="ten_b", model="m"), entry, ttl_seconds=60)

    assert await cache.clear_tenant("ten_a") == 1


async def test_a_failed_purge_is_reported_not_raised() -> None:
    """GDPR erasure continues across the other stores even if one is unavailable."""
    cache = RedisCache(redis=FakeRedis(fail=True))

    assert await cache.clear_tenant("ten_a") == 0
    assert cache.stats.errors == 1


async def test_a_failed_delete_does_not_raise() -> None:
    """Eviction is best-effort; the entry will expire on its own."""
    cache = RedisCache(redis=FakeRedis(fail=True))

    await cache.delete("k")

    assert cache.stats.errors == 1


# ── Tool result cache ────────────────────────────────────────────────────────


async def test_deterministic_tools_are_cached_for_a_long_time() -> None:
    """A calculator result does not go stale."""
    cache = ToolResultCache(InMemoryCache())

    await cache.set(tool="calculator", arguments={"expr": "2+2"}, tenant_id="t", result=4)

    assert await cache.get(tool="calculator", arguments={"expr": "2+2"}, tenant_id="t") == 4


async def test_code_execution_is_never_cached() -> None:
    """Side effects and non-determinism make a cached result actively wrong."""
    cache = ToolResultCache(InMemoryCache())

    await cache.set(tool="code_exec", arguments={"code": "x=1"}, tenant_id="t", result="ok")

    assert await cache.get(tool="code_exec", arguments={"code": "x=1"}, tenant_id="t") is None


async def test_tool_results_are_scoped_per_tenant() -> None:
    """A SQL result for one tenant must never be served to another."""
    cache = ToolResultCache(InMemoryCache())
    await cache.set(tool="sql_analytics", arguments={"q": "x"}, tenant_id="ten_a", result=[1])

    assert await cache.get(tool="sql_analytics", arguments={"q": "x"}, tenant_id="ten_b") is None


async def test_different_arguments_are_different_entries() -> None:
    """Otherwise every call to a tool returns the first result it ever produced."""
    cache = ToolResultCache(InMemoryCache())
    await cache.set(tool="calculator", arguments={"expr": "2+2"}, tenant_id="t", result=4)

    assert await cache.get(tool="calculator", arguments={"expr": "3+3"}, tenant_id="t") is None


def test_unknown_tools_get_the_conservative_default_ttl() -> None:
    """A new tool must not accidentally inherit the calculator's day-long TTL."""
    assert ToolResultCache(InMemoryCache()).ttl_for("brand_new_tool") == 300


# ── Semantic cache: the dangerous part ───────────────────────────────────────


@pytest.mark.parametrize(
    ("first", "second", "differs"),
    [
        ("how do I enable SSO", "how do I disable SSO", True),
        ("show me 5 results", "show me 10 results", True),
        ("which is the cheapest plan", "which is the most expensive plan", True),
        ("what happens before renewal", "what happens after renewal", True),
        ("what is our refund window", "how long is the refund period", False),
        ("who owns billing", "who is responsible for billing", False),
    ],
)
def test_polarity_and_quantity_differences_are_detected(
    first: str, second: str, differs: bool
) -> None:
    """These pairs embed almost identically and have opposite answers."""
    assert differs_by_negation(first, second) is differs


async def test_semantically_equivalent_query_hits() -> None:
    """The whole point: different words, same question, one answer."""
    vectors = {
        "how long is the refund window": [1.0, 0.0, 0.0],
        "what is the refund period": [1.0, 0.0, 0.0],
    }
    cache = SemanticCache(
        cache=InMemoryCache(), embedder=StaticEmbedder(vectors), router=verify_router("YES")
    )
    await cache.store("how long is the refund window", "Thirty days.", tenant_id="t")

    hit = await cache.lookup("what is the refund period", tenant_id="t")

    assert hit is not None
    assert hit.entry.value == "Thirty days."


async def test_an_opposite_question_is_rejected_despite_high_similarity() -> None:
    """The failure that makes naive semantic caching dangerous."""
    vectors = {"how do I enable SSO": [1.0, 0.0, 0.0], "how do I disable SSO": [1.0, 0.0, 0.0]}
    cache = SemanticCache(cache=InMemoryCache(), embedder=StaticEmbedder(vectors))
    await cache.store("how do I enable SSO", "Turn it on in settings.", tenant_id="t")

    hit = await cache.lookup("how do I disable SSO", tenant_id="t")

    assert hit is None
    assert cache.stats.false_hits == 1


async def test_a_dissimilar_query_misses() -> None:
    """Below the similarity floor there is nothing to verify."""
    vectors = {"refund policy": [1.0, 0.0, 0.0], "office locations": [0.0, 1.0, 0.0]}
    cache = SemanticCache(cache=InMemoryCache(), embedder=StaticEmbedder(vectors))
    await cache.store("refund policy", "Thirty days.", tenant_id="t")

    assert await cache.lookup("office locations", tenant_id="t") is None


async def test_llm_verification_can_veto_a_high_similarity_candidate() -> None:
    """The last line of defence, for pairs the lexical guards cannot separate."""
    vectors = {"revenue in EMEA": [1.0, 0.0, 0.0], "revenue in APAC": [1.0, 0.0, 0.0]}
    cache = SemanticCache(
        cache=InMemoryCache(), embedder=StaticEmbedder(vectors), router=verify_router("NO")
    )
    await cache.store("revenue in EMEA", "4 million euro", tenant_id="t")

    assert await cache.lookup("revenue in APAC", tenant_id="t") is None
    assert cache.stats.false_hits == 1


async def test_verification_outage_misses_rather_than_guessing() -> None:
    """Verification is what makes this cache safe; without it, do not serve."""
    pricing.MODEL_PROVIDERS["cache-dead-model"] = "fake"
    router = LLMRouter(
        providers={"fake": FakeProvider(fail_times=99)},
        policy=ModelPolicy(default_model="cache-dead-model"),
    )
    vectors = {"a question here": [1.0, 0.0, 0.0], "another question": [1.0, 0.0, 0.0]}
    cache = SemanticCache(cache=InMemoryCache(), embedder=StaticEmbedder(vectors), router=router)
    await cache.store("a question here", "answer", tenant_id="t")

    assert await cache.lookup("another question", tenant_id="t") is None


async def test_an_identical_query_skips_verification() -> None:
    """There is nothing to verify, and paying for it would be waste."""
    router = verify_router("NO")
    vectors = {"exact question": [1.0, 0.0, 0.0]}
    cache = SemanticCache(cache=InMemoryCache(), embedder=StaticEmbedder(vectors), router=router)
    await cache.store("exact question", "answer", tenant_id="t")

    hit = await cache.lookup("exact question", tenant_id="t")

    assert hit is not None
    assert router.stats.calls == 0


async def test_tenants_have_separate_semantic_indexes() -> None:
    """Cross-tenant reuse would be a data leak, not a cache hit."""
    vectors = {"shared question": [1.0, 0.0, 0.0]}
    cache = SemanticCache(cache=InMemoryCache(), embedder=StaticEmbedder(vectors))
    await cache.store("shared question", "tenant A answer", tenant_id="ten_a")

    assert await cache.lookup("shared question", tenant_id="ten_b") is None


async def test_ingestion_invalidates_the_tenant_cache() -> None:
    """Once the corpus changes, a cached answer may no longer match the documents."""
    vectors = {"question": [1.0, 0.0, 0.0]}
    cache = SemanticCache(cache=InMemoryCache(), embedder=StaticEmbedder(vectors))
    await cache.store("question", "old answer", tenant_id="t")

    await cache.invalidate_tenant("t")

    assert await cache.lookup("question", tenant_id="t") is None


async def test_an_empty_cache_misses_without_embedding_anything() -> None:
    """A cold cache must not pay to embed a query it cannot possibly match."""
    calls: list[str] = []

    class CountingEmbedder(StaticEmbedder):
        async def embed_query(self, text: str) -> list[float]:
            calls.append(text)
            return await super().embed_query(text)

    cache = SemanticCache(cache=InMemoryCache(), embedder=CountingEmbedder({}))

    assert await cache.lookup("anything", tenant_id="t") is None
    assert calls == []


async def test_entries_from_a_different_embedding_model_are_skipped() -> None:
    """Comparing vectors of different dimensions is meaningless, not a near miss."""
    cache = SemanticCache(cache=InMemoryCache(), embedder=StaticEmbedder({"q": [1.0, 0.0, 0.0]}))
    cache._index["t"] = [("stale-key", (1.0, 0.0), "old query")]

    assert await cache.lookup("q", tenant_id="t") is None


async def test_an_evicted_entry_is_removed_from_the_index() -> None:
    """A stale index row would make every later lookup pay for a guaranteed miss."""
    vectors = {"question": [1.0, 0.0, 0.0]}
    backend = InMemoryCache()
    cache = SemanticCache(cache=backend, embedder=StaticEmbedder(vectors))
    key = await cache.store("question", "answer", tenant_id="t")
    await backend.delete(key)

    assert await cache.lookup("question", tenant_id="t") is None
    assert cache._index["t"] == []


@pytest.mark.parametrize("threshold", [0.4, 1.1])
def test_nonsensical_similarity_thresholds_are_rejected(threshold: float) -> None:
    """A 0.4 threshold would match almost anything; that is never intended."""
    with pytest.raises(ValueError, match="threshold"):
        SemanticCache(cache=InMemoryCache(), embedder=StaticEmbedder({}), threshold=threshold)


def test_normalise_collapses_case_and_whitespace() -> None:
    """Comparison must not depend on formatting."""
    assert normalise("  How  LONG is it? ") == "how long is it?"


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [([1.0, 0.0], [1.0, 0.0], 1.0), ([1.0, 0.0], [0.0, 1.0], 0.0), ([0.0, 0.0], [1.0, 0.0], 0.0)],
)
def test_cosine_similarity(first: list[float], second: list[float], expected: float) -> None:
    """Zero vectors are maximally dissimilar rather than an error."""
    assert cosine_similarity(first, second) == pytest.approx(expected)


def test_mismatched_vector_lengths_raise() -> None:
    """Silently comparing a prefix would produce a plausible, wrong similarity."""
    with pytest.raises(ValueError, match="length mismatch"):
        cosine_similarity([1.0, 0.0], [1.0])
