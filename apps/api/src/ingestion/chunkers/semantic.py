"""Semantic chunking.

Instead of cutting at a fixed size, cut where the topic actually changes. The
document is split into sentences, each sentence is embedded, and a boundary is
placed wherever the cosine distance between consecutive sentence windows spikes
above a percentile threshold of the document's own distance distribution.

Using a *relative* threshold matters: an absolute one behaves completely
differently on a dense technical paper than on a chatty handbook, which is
exactly the failure mode that makes naive semantic chunking look bad in
benchmarks.

The embedder is injected rather than imported so this module stays pure and the
tests can drive it with a deterministic fake.

Example:
    >>> chunker = SemanticChunker(embed=lambda texts: [[float(len(t))] for t in texts])
    >>> chunker.name
    'semantic'
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from src.ingestion.chunkers.base import (
    normalise_whitespace,
    register_chunker,
    split_oversized_text,
    split_sentences,
)
from src.ingestion.types import ChunkDraft, ChunkingConfig, ParsedDocument
from src.models.document import ChunkKind

if TYPE_CHECKING:
    from src.ingestion.types import TextBlock

#: Signature of the injected embedding function.
EmbedFn = Callable[[list[str]], Sequence[Sequence[float]]]

#: Sentences either side of a candidate boundary that are averaged before
#: comparing. A window of 1 is far too jittery on real prose.
_WINDOW = 2


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine distance in [0, 2], with zero vectors treated as maximally distant.

    Example:
        >>> cosine_distance([1.0, 0.0], [1.0, 0.0])
        0.0
        >>> round(cosine_distance([1.0, 0.0], [0.0, 1.0]), 6)
        1.0
        >>> cosine_distance([0.0, 0.0], [1.0, 0.0])
        1.0
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of ``values``, with ``q`` in [0, 100].

    Implemented here rather than pulled from numpy so the chunker has no heavy
    import and behaves identically in the worker and in a property test.

    Example:
        >>> percentile([1.0, 2.0, 3.0, 4.0], 50)
        2.5
        >>> percentile([5.0], 95)
        5.0
    """
    if not values:
        msg = "percentile of an empty sequence is undefined"
        raise ValueError(msg)
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * min(max(q, 0.0), 100.0) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class SemanticChunker:
    """Cuts a document where its meaning shifts rather than where the byte count says to."""

    name = "semantic"

    def __init__(self, *, embed: EmbedFn, breakpoint_percentile: float = 90.0) -> None:
        """Create a semantic chunker.

        Args:
            embed: Function mapping a list of texts to their embeddings. Must be
                deterministic; batching is the caller's concern.
            breakpoint_percentile: Distances above this percentile of the
                document's own distribution become chunk boundaries. Higher means
                fewer, larger chunks.
        """
        if not 50.0 <= breakpoint_percentile <= 99.9:
            msg = "breakpoint_percentile must be between 50 and 99.9"
            raise ValueError(msg)
        self._embed = embed
        self._breakpoint_percentile = breakpoint_percentile

    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> list[ChunkDraft]:
        """Split ``document`` at semantic boundaries, respecting the size ceiling.

        Atomic blocks (tables, code) bypass the semantic pass entirely — their
        boundaries are structural, not topical.
        """
        drafts: list[ChunkDraft] = []
        for block in document.blocks:
            if not block.text.strip():
                continue
            if block.is_atomic:
                drafts.append(self._draft(block.text, block, len(drafts), config))
                continue
            for group in self._semantic_groups(block.text, config):
                drafts.append(self._draft(group, block, len(drafts), config))
        return drafts

    # ── internals ────────────────────────────────────────────────────────────

    def _semantic_groups(self, text: str, config: ChunkingConfig) -> list[str]:
        """Group a block's sentences into semantically coherent runs."""
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return split_oversized_text(
                normalise_whitespace(text),
                max_chars=config.max_chars,
                overlap_chars=config.overlap_chars,
            )

        boundaries = self._boundary_indices(sentences)
        groups: list[str] = []
        current: list[str] = []
        for index, sentence in enumerate(sentences):
            candidate_len = sum(len(s) + 1 for s in current) + len(sentence)
            # A semantic boundary, or the hard size ceiling, closes the group.
            if current and (index in boundaries or candidate_len > config.max_chars):
                groups.append(" ".join(current))
                current = []
            current.append(sentence)
        if current:
            groups.append(" ".join(current))

        # A group may still exceed the ceiling if one sentence is enormous.
        final: list[str] = []
        for group in groups:
            final.extend(
                split_oversized_text(group, max_chars=config.max_chars, overlap_chars=0)
                if len(group) > config.max_chars
                else [group]
            )
        return final

    def _boundary_indices(self, sentences: list[str]) -> frozenset[int]:
        """Indices at which a new chunk should begin."""
        embeddings = list(self._embed(list(sentences)))
        if len(embeddings) != len(sentences):
            msg = "embed() returned a different number of vectors than sentences"
            raise ValueError(msg)

        distances: list[float] = []
        for index in range(1, len(sentences)):
            before = _mean_vector(embeddings[max(0, index - _WINDOW) : index])
            after = _mean_vector(embeddings[index : index + _WINDOW])
            distances.append(cosine_distance(before, after))

        if not distances:
            return frozenset()
        threshold = percentile(distances, self._breakpoint_percentile)
        # `distances[i]` is the gap *before* sentence i+1.
        return frozenset(
            index + 1 for index, distance in enumerate(distances) if distance >= threshold
        )

    def _draft(
        self, body: str, block: TextBlock, ordinal: int, config: ChunkingConfig
    ) -> ChunkDraft:
        """Wrap a group of text as a chunk draft carrying the block's provenance."""
        content = normalise_whitespace(body)
        if config.prepend_section_path and block.section_path:
            content = f"{' > '.join(block.section_path)}\n\n{content}"
        return ChunkDraft(
            content=content,
            ordinal=ordinal,
            kind=block.kind if block.kind is not ChunkKind.HEADING else ChunkKind.PROSE,
            page_number=block.page_number,
            section_path=block.section_path,
            bbox=block.bbox,
            metadata={"strategy": self.name, "breakpoint_pct": self._breakpoint_percentile},
        )


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Element-wise mean of a non-empty sequence of equal-length vectors."""
    if not vectors:
        msg = "cannot average an empty window"
        raise ValueError(msg)
    width = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(width)]


def build_semantic_chunker(
    embed: EmbedFn, *, breakpoint_percentile: float = 90.0
) -> SemanticChunker:
    """Construct and register a semantic chunker bound to a concrete embedder.

    Registration happens here rather than at import time because the strategy is
    useless without an embedding function, and the embedder depends on the
    tenant's configured model.
    """
    chunker = SemanticChunker(embed=embed, breakpoint_percentile=breakpoint_percentile)
    register_chunker(chunker)
    return chunker
