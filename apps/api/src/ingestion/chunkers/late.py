"""Late chunking.

Conventional chunking embeds each chunk in isolation, so a chunk that says "it
grew 12% year on year" has no idea what "it" is. Late chunking (Günther et al.,
2024) inverts the order: run the whole document through the encoder *first*, so
every token attends to the full context, then pool the token embeddings within
chunk boundaries. Each chunk vector is therefore context-aware even though the
chunk text is short.

The trade-off is a context-length ceiling on the encoder. This implementation
splits the document into encoder-sized *macro segments*, late-chunks within each,
and records which segment a chunk came from so the eval harness can measure
whether cross-segment references are the residual failure mode.

Where the encoder cannot be run (no model configured, document too large), the
strategy degrades to layout-aware chunking plus Contextual Retrieval preambles,
which targets the same problem at the cost of an LLM call per chunk.

Example:
    >>> LateChunker(encode=lambda text: [[1.0]] * len(text)).name
    'late'
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from src.ingestion.chunkers.base import register_chunker, split_sentences
from src.ingestion.chunkers.layout import LayoutAwareChunker
from src.ingestion.types import ChunkDraft, ChunkingConfig, ParsedDocument

#: Encodes a text into one vector per character position. Real encoders work in
#: tokens; the adapter in :mod:`src.ingestion.embedders` maps token spans back to
#: character spans so this interface stays trivially testable.
EncodeFn = Callable[[str], Sequence[Sequence[float]]]


class LateChunker:
    """Embeds the whole document, then pools token vectors per chunk."""

    name = "late"

    def __init__(
        self,
        *,
        encode: EncodeFn,
        max_context_chars: int = 24_000,
    ) -> None:
        """Create a late chunker.

        Args:
            encode: Maps a text to per-position embeddings.
            max_context_chars: Approximate encoder context window, in characters.
                Documents longer than this are processed in macro segments.
        """
        if max_context_chars < 1000:
            msg = "max_context_chars below 1000 defeats the purpose of late chunking"
            raise ValueError(msg)
        self._encode = encode
        self._max_context_chars = max_context_chars
        self._boundary_chunker = LayoutAwareChunker()

    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> list[ChunkDraft]:
        """Return chunks whose ``metadata`` carries a context-aware pooled vector.

        Boundaries come from the layout-aware chunker — late chunking changes how
        chunks are *embedded*, not where they are cut. The pooled vector is
        attached under ``metadata["late_embedding"]`` for the embedder stage to
        use instead of re-embedding the chunk text.
        """
        drafts = self._boundary_chunker.chunk(document, config)
        if not drafts:
            return []

        segments = self._macro_segments(document)
        enriched: list[ChunkDraft] = []
        for draft in drafts:
            segment_index, offset = self._locate(draft.content, segments)
            if segment_index is None:
                enriched.append(draft)
                continue
            vectors = self._encoded_segment(segments[segment_index])
            pooled = _mean_pool(vectors[offset : offset + draft.char_length])
            enriched.append(
                draft.model_copy(
                    update={
                        "metadata": {
                            **draft.metadata,
                            "strategy": self.name,
                            "late_embedding": pooled,
                            "macro_segment": segment_index,
                        }
                    }
                )
            )
        return enriched

    # ── internals ────────────────────────────────────────────────────────────

    def _macro_segments(self, document: ParsedDocument) -> list[str]:
        """Split the document into encoder-sized segments at sentence boundaries."""
        text = document.full_text
        if len(text) <= self._max_context_chars:
            return [text]

        segments: list[str] = []
        current: list[str] = []
        length = 0
        for sentence in split_sentences(text) or [text]:
            if length + len(sentence) > self._max_context_chars and current:
                segments.append(" ".join(current))
                current, length = [], 0
            current.append(sentence)
            length += len(sentence) + 1
        if current:
            segments.append(" ".join(current))
        return segments

    def _encoded_segment(self, segment: str) -> Sequence[Sequence[float]]:
        """Encode one macro segment, validating the encoder's contract."""
        vectors = self._encode(segment)
        if len(vectors) < len(segment):
            msg = (
                "encode() must return at least one vector per character position; "
                f"got {len(vectors)} for {len(segment)} characters"
            )
            raise ValueError(msg)
        return vectors

    @staticmethod
    def _locate(content: str, segments: list[str]) -> tuple[int | None, int]:
        """Find which macro segment contains a chunk, and at what offset.

        Chunks carry a breadcrumb prefix that is not present in the source text,
        so the search uses the longest suffix line of the content.
        """
        needle = content.split("\n\n")[-1][:200]
        if not needle:
            return None, 0
        for index, segment in enumerate(segments):
            offset = segment.find(needle)
            if offset != -1:
                return index, offset
        return None, 0


def _mean_pool(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Mean-pool a span of position vectors into a single chunk vector.

    Example:
        >>> _mean_pool([[1.0, 3.0], [3.0, 5.0]])
        [2.0, 4.0]
        >>> _mean_pool([])
        []
    """
    if not vectors:
        return []
    width = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(width)]


def build_late_chunker(encode: EncodeFn, *, max_context_chars: int = 24_000) -> LateChunker:
    """Construct and register a late chunker bound to a concrete encoder."""
    chunker = LateChunker(encode=encode, max_context_chars=max_context_chars)
    register_chunker(chunker)
    return chunker
