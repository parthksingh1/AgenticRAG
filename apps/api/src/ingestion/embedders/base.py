"""Embedding models and batching.

Embedding is the expensive, rate-limited part of ingestion, so this module cares
about three things beyond "call the model":

* **Deduplication.** Identical text is embedded once per batch. Real corpora are
  full of repeated boilerplate — headers, footers, legal notices — and skipping
  the duplicates typically removes a meaningful fraction of the work for free.
* **Dynamic batch sizing.** Batch size adapts to the available memory rather
  than being a constant that is wrong on every machine except the author's.
* **Determinism in tests.** :class:`HashingEmbedder` produces stable vectors with
  no model download, so the ingestion pipeline is fully testable offline.

Example:
    >>> import asyncio
    >>> embedder = HashingEmbedder(dimension=8)
    >>> vectors = asyncio.run(embedder.embed(["hello", "hello", "world"]))
    >>> vectors[0] == vectors[1] and vectors[0] != vectors[2]
    True
"""

from __future__ import annotations

import hashlib
import math
import struct
from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.core.logging import get_logger

log = get_logger(__name__)

#: Texts longer than this are truncated before embedding. Every supported model
#: has a shorter window than this, and truncating loudly beats a provider 400.
MAX_EMBED_CHARS = 32_000


class Embedder(ABC):
    """Turns text into dense vectors."""

    #: Model identifier stored on the chunk, so a later dimension change is
    #: detectable rather than a silent similarity collapse.
    model_name: str
    dimension: int

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input, in order."""

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text.

        Example:
            >>> import asyncio
            >>> len(asyncio.run(HashingEmbedder(dimension=4).embed_one("x")))
            4
        """
        return (await self.embed([text]))[0]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        Split from :meth:`embed` because asymmetric models (BGE, E5) require a
        different prefix for queries than for documents, and using the document
        form for queries silently costs several points of recall.
        """
        return await self.embed_one(text)


def deduplicate(texts: Sequence[str]) -> tuple[list[str], list[int]]:
    """Collapse repeated texts, returning the unique list and an index map.

    Returns:
        ``(unique, mapping)`` where ``mapping[i]`` is the index into ``unique``
        for input ``i``. Rehydrating is ``[vectors[j] for j in mapping]``.

    Example:
        >>> deduplicate(["a", "b", "a"])
        (['a', 'b'], [0, 1, 0])
    """
    seen: dict[str, int] = {}
    unique: list[str] = []
    mapping: list[int] = []
    for text in texts:
        index = seen.get(text)
        if index is None:
            index = len(unique)
            seen[text] = index
            unique.append(text)
        mapping.append(index)
    return unique, mapping


def dynamic_batch_size(
    texts: Sequence[str],
    *,
    max_batch: int = 64,
    min_batch: int = 1,
    target_chars_per_batch: int = 60_000,
) -> int:
    """Pick a batch size from the actual text sizes.

    A fixed batch size is either wasteful on short chunks or an out-of-memory
    error on long ones. Sizing by total characters keeps each batch's memory
    footprint roughly constant regardless of chunk length.

    Example:
        >>> dynamic_batch_size(["short"] * 100)
        64
        >>> dynamic_batch_size(["x" * 30_000] * 10)
        2
        >>> dynamic_batch_size([])
        1
    """
    if not texts:
        return min_batch
    mean_chars = max(sum(len(t) for t in texts) / len(texts), 1.0)
    estimated = int(target_chars_per_batch / mean_chars)
    return max(min_batch, min(max_batch, estimated))


def truncate(text: str, *, limit: int = MAX_EMBED_CHARS) -> str:
    """Cut text to the embedding limit, warning when anything is lost.

    Example:
        >>> truncate("abcdef", limit=3)
        'abc'
    """
    if len(text) <= limit:
        return text
    log.warning("truncating text before embedding", original_chars=len(text), limit=limit)
    return text[:limit]


class HashingEmbedder(Embedder):
    """Deterministic, dependency-free embedder for tests and offline demos.

    Produces a normalised vector from a hash of the text's token trigrams. It is
    not semantically meaningful, but it *is* stable, cheap and identical across
    machines — which is exactly what the ingestion tests and the offline demo
    need. Anything that needs real semantics uses a real model.
    """

    def __init__(self, *, dimension: int = 1024, model_name: str = "hashing-test-embedder") -> None:
        """Create a hashing embedder of the given dimension."""
        if dimension < 2:
            msg = "dimension must be at least 2"
            raise ValueError(msg)
        self.dimension = dimension
        self.model_name = model_name

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts by hashing, deduplicating identical inputs."""
        unique, mapping = deduplicate([truncate(t) for t in texts])
        vectors = [self._vector(text) for text in unique]
        return [vectors[i] for i in mapping]

    def _vector(self, text: str) -> list[float]:
        """Hash one text into a unit-norm vector."""
        vector = [0.0] * self.dimension
        tokens = text.lower().split() or [""]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = struct.unpack("<Q", digest)[0] % self.dimension
            # Sign from a second hash so buckets can cancel, which spreads the
            # distribution instead of piling every text into the positive orthant.
            sign = 1.0 if digest[0] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class SentenceTransformerEmbedder(Embedder):
    """Wraps a sentence-transformers model (BGE, E5, or any local encoder).

    The model is loaded lazily on first use so that importing the module — which
    the API does at startup — does not pull hundreds of megabytes into memory in
    a process that may never embed anything.

    Encoding is CPU/GPU-bound, so it runs in a thread to avoid blocking the event
    loop; a synchronous ``model.encode`` inside an async handler would stall every
    other request on the worker.
    """

    #: Instruction prefixes required by asymmetric models. Using the document
    #: form for a query measurably costs recall, so this is not cosmetic.
    QUERY_PREFIXES = {  # noqa: RUF012 - class-level config table
        "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
        "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
        "intfloat/multilingual-e5-large": "query: ",
    }
    DOCUMENT_PREFIXES = {  # noqa: RUF012 - class-level config table
        "intfloat/multilingual-e5-large": "passage: ",
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        *,
        dimension: int = 1024,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        """Configure the embedder, verifying the dependency is importable.

        The *model* is still loaded lazily — that is the slow part, and a worker
        should not pay for it at import time. But the **import is checked here**,
        because every caller wraps construction in a try/except to fall back to
        the hashing embedder, and a constructor that cannot fail defeats all of
        them: the ModuleNotFoundError then surfaces mid-ingestion, after the
        document row exists and the user has been told their upload is being
        processed.

        Raises:
            ImportError: when sentence-transformers is not installed.
        """
        import importlib.util

        if importlib.util.find_spec("sentence_transformers") is None:
            msg = (
                "sentence-transformers is not installed; install it or use "
                "HashingEmbedder for a degraded but working pipeline"
            )
            raise ImportError(msg)

        self.model_name = model_name
        self.dimension = dimension
        self._device = device
        self._normalize = normalize
        self._model: object | None = None

    def _load(self) -> object:
        """Load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model", model=self.model_name, device=self._device)
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents, batching and deduplicating."""
        prefix = self.DOCUMENT_PREFIXES.get(self.model_name, "")
        return await self._encode([f"{prefix}{truncate(t)}" for t in texts])

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query with the model's query instruction prefix."""
        prefix = self.QUERY_PREFIXES.get(self.model_name, "")
        return (await self._encode([f"{prefix}{truncate(text)}"]))[0]

    async def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Run the encoder off the event loop, in memory-sized batches."""
        import asyncio

        unique, mapping = deduplicate(list(texts))
        model = self._load()
        batch_size = dynamic_batch_size(unique)

        vectors: list[list[float]] = []
        for start in range(0, len(unique), batch_size):
            window = unique[start : start + batch_size]
            encoded = await asyncio.to_thread(
                model.encode,  # type: ignore[attr-defined]
                window,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            vectors.extend([float(v) for v in row] for row in encoded)

        return [vectors[i] for i in mapping]


class CachingEmbedder(Embedder):
    """Wraps another embedder with a content-addressed cache.

    Re-ingesting a document version whose text is unchanged should cost nothing,
    and near-duplicate documents across a tenant share most of their boilerplate.
    The cache is keyed by ``(model, text)`` so switching models cannot serve a
    vector from the wrong space — the most dangerous possible cache bug here,
    since it fails silently as slightly worse retrieval rather than as an error.
    """

    def __init__(self, inner: Embedder, *, cache: EmbeddingCache) -> None:
        """Wrap ``inner``, reading and writing through ``cache``."""
        self._inner = inner
        self._cache = cache
        self.model_name = inner.model_name
        self.dimension = inner.dimension
        self.hits = 0
        self.misses = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed, serving what the cache already has."""
        keys = [self._key(t) for t in texts]
        cached = await self._cache.get_many(keys)

        missing_indices = [i for i, key in enumerate(keys) if cached.get(key) is None]
        self.hits += len(texts) - len(missing_indices)
        self.misses += len(missing_indices)

        if missing_indices:
            fresh = await self._inner.embed([texts[i] for i in missing_indices])
            await self._cache.set_many(
                {keys[i]: vector for i, vector in zip(missing_indices, fresh, strict=True)}
            )
            for i, vector in zip(missing_indices, fresh, strict=True):
                cached[keys[i]] = vector

        return [cached[key] for key in keys]  # type: ignore[misc]

    async def embed_query(self, text: str) -> list[float]:
        """Queries bypass the cache: they are rarely repeated verbatim."""
        return await self._inner.embed_query(text)

    def _key(self, text: str) -> str:
        """Content-addressed cache key, namespaced by model."""
        digest = hashlib.sha256(f"{self.model_name}\x00{text}".encode()).hexdigest()
        return f"emb:{digest}"

    @property
    def hit_ratio(self) -> float:
        """Fraction of embed requests served from cache.

        Example:
            >>> from src.ingestion.embedders.base import CachingEmbedder, InMemoryEmbeddingCache
            >>> e = CachingEmbedder(HashingEmbedder(dimension=4), cache=InMemoryEmbeddingCache())
            >>> e.hit_ratio
            0.0
        """
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class EmbeddingCache(ABC):
    """Storage for computed embeddings."""

    @abstractmethod
    async def get_many(self, keys: Sequence[str]) -> dict[str, list[float] | None]:
        """Return cached vectors, with None for misses."""

    @abstractmethod
    async def set_many(self, entries: dict[str, list[float]]) -> None:
        """Store computed vectors."""


class InMemoryEmbeddingCache(EmbeddingCache):
    """Process-local embedding cache, used in tests and by the CLI.

    Production uses the Redis-backed cache in :mod:`src.caching`; this exists so
    the ingestion pipeline can be exercised without a running Redis.
    """

    def __init__(self) -> None:
        """Create an empty cache."""
        self._store: dict[str, list[float]] = {}

    async def get_many(self, keys: Sequence[str]) -> dict[str, list[float] | None]:
        """Look up many keys at once."""
        return {key: self._store.get(key) for key in keys}

    async def set_many(self, entries: dict[str, list[float]]) -> None:
        """Store many vectors at once."""
        self._store.update(entries)
