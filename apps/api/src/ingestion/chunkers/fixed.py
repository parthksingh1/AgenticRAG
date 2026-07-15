"""Fixed-size chunking.

The naive baseline: cut every ``target_chars`` at the nearest sentence boundary,
ignoring layout entirely. It exists so the eval harness can quantify what
layout-aware and semantic chunking actually buy — a claim like "layout-aware
chunking improves context_precision by N points" is only meaningful against a
baseline that is in the repository and runnable.

Example:
    >>> from src.ingestion.types import ChunkingConfig, ParsedDocument, TextBlock
    >>> doc = ParsedDocument(blocks=(TextBlock(text="One. Two. Three."),))
    >>> [c.content for c in FixedSizeChunker().chunk(doc, ChunkingConfig(target_chars=100))]
    ['One. Two. Three.']
"""

from __future__ import annotations

from src.ingestion.chunkers.base import (
    apply_overlap,
    normalise_whitespace,
    register_chunker,
    split_oversized_text,
)
from src.ingestion.types import ChunkDraft, ChunkingConfig, ParsedDocument
from src.models.document import ChunkKind


class FixedSizeChunker:
    """Splits the document's text into equal-sized, overlapping windows."""

    name = "fixed"

    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> list[ChunkDraft]:
        """Split ``document`` into fixed-size chunks with overlap.

        Structure is deliberately discarded: no breadcrumbs, no atomic-block
        protection, no section awareness. That is the point of a baseline.
        """
        text = normalise_whitespace(document.full_text)
        if not text:
            return []

        pieces = split_oversized_text(text, max_chars=config.target_chars, overlap_chars=0)
        drafts: list[ChunkDraft] = []
        previous = ""
        for piece in pieces:
            body = piece.strip()
            if not body:
                continue
            content = apply_overlap(previous, body, overlap_chars=config.overlap_chars)
            drafts.append(
                ChunkDraft(
                    content=content,
                    ordinal=len(drafts),
                    kind=ChunkKind.PROSE,
                    metadata={"strategy": self.name},
                )
            )
            previous = body
        return drafts


fixed_size_chunker = register_chunker(FixedSizeChunker())
