"""Caching layer: exact, semantic, tool-result and embedding caches."""

from src.caching.base import (
    Cache,
    CacheEntry,
    CacheStats,
    InMemoryCache,
    RedisCache,
    ToolResultCache,
    build_cache_key,
    cosine_similarity,
)
from src.caching.semantic import SemanticCache, SemanticHit, differs_by_negation, normalise

__all__ = [
    "Cache",
    "CacheEntry",
    "CacheStats",
    "InMemoryCache",
    "RedisCache",
    "SemanticCache",
    "SemanticHit",
    "ToolResultCache",
    "build_cache_key",
    "cosine_similarity",
    "differs_by_negation",
    "normalise",
]
