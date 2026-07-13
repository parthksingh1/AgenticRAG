"""Semantic answer cache.

An exact cache only helps when two users type the same characters, which is
rare. A semantic cache matches on meaning, so "what's our refund window?" reuses
the answer to "how long do customers have to return something?" — a much larger
share of real traffic.

The danger is the reason most semantic caches are a bad idea in practice: near
neighbours in embedding space are not the same question. "How do I *enable* SSO?"
and "How do I *disable* SSO?" have cosine similarity well above 0.95 and opposite
answers. Serving the wrong one is worse than any latency saving.

Three defences, in order of cost:

1. **A high similarity floor** (0.97 by default). Not sufficient on its own —
   the SSO example clears it — but it removes the obvious mismatches for free.
2. **Negation and quantifier guarding.** Questions differing only by a negation,
   a comparative or a number are never treated as equivalent, no matter how
   similar they embed. This is cheap and catches the specific failure that makes
   semantic caching dangerous.
3. **A cheap LLM verification** of the surviving candidate. Only reached for
   entries that passed the first two, so it runs on a small fraction of lookups.

Every rejected hit is counted as a ``false_hit``, which makes the cache's real
accuracy measurable rather than assumed.

Example:
    >>> differs_by_negation("how to enable SSO", "how to disable SSO")
    True
    >>> differs_by_negation("what is our refund window", "how long is the refund period")
    False
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.caching.base import Cache, CacheEntry, CacheStats, cosine_similarity
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.ingestion.embedders.base import Embedder
    from src.services.llm.router import LLMRouter

log = get_logger(__name__)

#: Words whose presence or absence flips a question's meaning while barely
#: moving its embedding.
_POLARITY_TOKENS = frozenset(
    {
        "not",
        "no",
        "never",
        "without",
        "except",
        "excluding",
        "disable",
        "disabled",
        "enable",
        "enabled",
        "remove",
        "add",
        "revoke",
        "grant",
        "cannot",
        "can't",
        "don't",
        "doesn't",
        "isn't",
        "aren't",
        "stop",
        "start",
        "delete",
        "create",
        "increase",
        "decrease",
        "before",
        "after",
        "more",
        "less",
        "least",
        "most",
        "min",
        "max",
        "minimum",
        "maximum",
        "cheapest",
        "expensive",
        "oldest",
        "newest",
        "first",
        "last",
    }
)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")

_VERIFY_SYSTEM = (
    "You decide whether two questions are asking for exactly the same information, "
    "such that one answer serves both. Differences in wording, politeness or "
    "grammar do not matter. Differences in scope, entity, time period, quantity or "
    "polarity do matter. Reply with only YES or NO."
)


@dataclass(slots=True)
class SemanticHit:
    """A candidate match from the semantic cache."""

    entry: CacheEntry
    similarity: float
    key: str

    @property
    def source_query(self) -> str:
        """The query that originally produced the cached answer."""
        return self.entry.source_query or ""


def normalise(text: str) -> str:
    """Lowercase and collapse whitespace for comparison.

    Example:
        >>> normalise("  How  LONG is it? ")
        'how long is it?'
    """
    return " ".join(text.lower().split())


def differs_by_negation(a: str, b: str) -> bool:
    """Whether two questions differ in polarity, quantity or comparison.

    This is the guard that makes semantic caching safe enough to use. It is
    deliberately blunt: a false rejection costs one cache miss, whereas a false
    acceptance serves the opposite answer to the one asked for.

    Example:
        >>> differs_by_negation("can I enable SSO", "can I disable SSO")
        True
        >>> differs_by_negation("show 5 results", "show 10 results")
        True
        >>> differs_by_negation("what is the refund window", "how long is the refund window")
        False
    """
    tokens_a = {w.strip(".,?!;:") for w in normalise(a).split()}
    tokens_b = {w.strip(".,?!;:") for w in normalise(b).split()}

    if (tokens_a & _POLARITY_TOKENS) != (tokens_b & _POLARITY_TOKENS):
        return True
    return set(_NUMBER.findall(a)) != set(_NUMBER.findall(b))


class SemanticCache:
    """Answer cache keyed by query embedding."""

    def __init__(
        self,
        *,
        cache: Cache,
        embedder: Embedder,
        router: LLMRouter | None = None,
        verify_model: str | None = None,
        threshold: float = 0.97,
        ttl_seconds: int = 3600,
        max_candidates: int = 20,
    ) -> None:
        """Create the cache.

        Args:
            cache: Backend holding the entries.
            embedder: Embeds queries for comparison.
            router: Used for the verification call. Without it, verification
                falls back to the lexical guards alone.
            verify_model: Model for verification; the cheap model is right here.
            threshold: Minimum cosine similarity to consider a candidate.
            ttl_seconds: Entry lifetime.
            max_candidates: Vectors compared per lookup. The index is per tenant
                and small; beyond this the comparison costs more than the miss.
        """
        if not 0.5 <= threshold <= 1.0:
            msg = "threshold must be between 0.5 and 1.0"
            raise ValueError(msg)

        self._cache = cache
        self._embedder = embedder
        self._router = router
        self._verify_model = verify_model
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._max_candidates = max_candidates
        #: tenant -> [(key, vector, query)]. In production this is a small
        #: per-tenant vector set in Redis; here it is the same shape in memory.
        self._index: dict[str, list[tuple[str, tuple[float, ...], str]]] = {}
        self.stats = CacheStats()

    async def lookup(self, query: str, *, tenant_id: str) -> SemanticHit | None:
        """Find a cached answer for a semantically equivalent query.

        Returns None on a miss, or when a candidate was found but rejected by
        verification — in which case the rejection is counted as a false hit.
        """
        candidates = self._index.get(tenant_id)
        if not candidates:
            self.stats.misses += 1
            return None

        vector = await self._embedder.embed_query(query)
        best = self._best_candidate(vector, candidates)
        if best is None:
            self.stats.misses += 1
            return None

        key, similarity, source_query = best
        self.stats.similarities.append(similarity)

        # Cheap structural guard before spending anything on verification.
        if differs_by_negation(query, source_query):
            self.stats.false_hits += 1
            log.info(
                "semantic cache candidate rejected on polarity or quantity",
                similarity=round(similarity, 4),
            )
            return None

        entry = await self._cache.get(key)
        if entry is None or not entry.is_fresh(self._ttl):
            self.stats.misses += 1
            self._forget(tenant_id, key)
            return None

        if not await self._verify(query, source_query):
            self.stats.false_hits += 1
            return None

        self.stats.hits += 1
        log.info("semantic cache hit", similarity=round(similarity, 4))
        return SemanticHit(entry=entry, similarity=similarity, key=key)

    async def store(
        self,
        query: str,
        value: Any,
        *,
        tenant_id: str,
        model: str | None = None,
        prompt_version: str | None = None,
    ) -> str:
        """Cache an answer under its query embedding."""
        from src.caching.base import build_cache_key

        vector = await self._embedder.embed_query(query)
        key = build_cache_key(
            query=query,
            tenant_id=tenant_id,
            model=model or "",
            prompt_version=prompt_version,
            namespace="semantic",
        )
        await self._cache.set(
            key,
            CacheEntry(
                value=value,
                created_at=time.time(),
                model=model,
                prompt_version=prompt_version,
                source_query=query,
            ),
            ttl_seconds=self._ttl,
        )

        bucket = self._index.setdefault(tenant_id, [])
        bucket[:] = [row for row in bucket if row[0] != key]
        bucket.append((key, tuple(vector), query))
        # Bound the per-tenant index; oldest entries fall out first.
        if len(bucket) > self._max_candidates * 5:
            del bucket[: len(bucket) - self._max_candidates * 5]

        self.stats.writes += 1
        return key

    async def invalidate_tenant(self, tenant_id: str) -> int:
        """Drop every cached answer for a tenant.

        Called after ingestion as well as on erasure: once the corpus changes,
        a cached answer may no longer reflect what the documents say.
        """
        self._index.pop(tenant_id, None)
        return await self._cache.clear_tenant(tenant_id)

    def _best_candidate(
        self,
        vector: Sequence[float],
        candidates: Sequence[tuple[str, tuple[float, ...], str]],
    ) -> tuple[str, float, str] | None:
        """Highest-similarity candidate above the threshold, if any."""
        best: tuple[str, float, str] | None = None
        for key, cached_vector, source_query in candidates[-self._max_candidates * 5 :]:
            if len(cached_vector) != len(vector):
                continue  # a different embedding model wrote this entry
            similarity = cosine_similarity(vector, cached_vector)
            if similarity >= self._threshold and (best is None or similarity > best[1]):
                best = (key, similarity, source_query)
        return best

    def _forget(self, tenant_id: str, key: str) -> None:
        """Remove a stale key from the tenant's index."""
        bucket = self._index.get(tenant_id)
        if bucket:
            bucket[:] = [row for row in bucket if row[0] != key]

    async def _verify(self, query: str, source_query: str) -> bool:
        """Confirm two queries really want the same answer.

        Identical normalised text short-circuits — there is nothing to verify.
        Without a router, the lexical guards already applied are the whole check,
        which is recorded rather than silently assumed.
        """
        if normalise(query) == normalise(source_query):
            return True
        if self._router is None:
            return True

        from src.services.llm.types import CompletionRequest, Message

        request = CompletionRequest(
            messages=(
                Message.system(_VERIFY_SYSTEM),
                Message.user(f"Question A: {query}\nQuestion B: {source_query}"),
            ),
            model=self._verify_model or "",
            max_tokens=4,
            temperature=0.0,
            node="semantic_cache_verify",
        )
        try:
            completion = await self._router.complete(
                request.model_copy(update={"model": self._verify_model or request.model}),
                allow_fallback=False,
            )
        except Exception as exc:  # noqa: BLE001 - fail safe, not fast
            # Verification is the thing that makes this cache safe. If it cannot
            # run, treat the candidate as unverified and miss.
            log.warning("semantic cache verification unavailable; missing", reason=str(exc))
            return False

        return completion.content.strip().upper().startswith("YES")
