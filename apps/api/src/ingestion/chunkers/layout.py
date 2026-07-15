r"""Layout-aware chunking.

The default strategy. It packs parser blocks into chunks up to a target size,
subject to four rules that plain fixed-size splitting gets wrong:

1. **Atomic blocks stay whole.** A table or code listing is emitted as its own
   chunk even when it exceeds the target, because half a table retrieves as
   noise and renders as nonsense in the citation panel.
2. **Chunks do not straddle sections.** A heading flushes the current chunk, so
   a chunk never mixes "2.1 Setup" with "3 Results".
3. **Headings travel with their content** as a breadcrumb prefix rather than as
   body text, which disambiguates documents whose sections all read alike
   without duplicating the heading into the chunk twice.
4. **The size ceiling is real.** Breadcrumb and overlap are charged against the
   budget before packing, so ``len(chunk.content) <= max_chars`` holds for every
   emitted chunk. Atomic blocks are the single, explicitly flagged exception.

Example:
    >>> from src.ingestion.types import ChunkingConfig, ParsedDocument, TextBlock
    >>> doc = ParsedDocument(blocks=(TextBlock(text="Hello."), TextBlock(text="World.")))
    >>> [c.content for c in LayoutAwareChunker().chunk(doc, ChunkingConfig())]
    ['Hello.\n\nWorld.']
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.ingestion.chunkers.base import (
    apply_overlap,
    normalise_whitespace,
    register_chunker,
    split_oversized_text,
)
from src.ingestion.types import BoundingBox, ChunkDraft, ChunkingConfig, ParsedDocument, TextBlock
from src.models.document import ChunkKind

#: Separator between blocks packed into the same chunk.
_JOIN = "\n\n"

#: An atomic block may exceed ``max_chars`` by this factor before being split
#: anyway. Beyond it a single chunk would not fit an embedding context.
_ATOMIC_OVERSIZE_FACTOR = 2


@dataclass(slots=True)
class _Accumulator:
    """Mutable buffer of body blocks being packed into the current chunk."""

    blocks: list[TextBlock] = field(default_factory=list)

    @property
    def char_length(self) -> int:
        """Length of the joined buffer, including separators."""
        if not self.blocks:
            return 0
        return sum(b.char_length for b in self.blocks) + len(_JOIN) * (len(self.blocks) - 1)

    @property
    def is_empty(self) -> bool:
        """True when nothing has been buffered yet."""
        return not self.blocks

    def text(self) -> str:
        """Join buffered blocks into chunk body text."""
        return _JOIN.join(block.text for block in self.blocks)

    def clear(self) -> None:
        """Drop everything buffered."""
        self.blocks.clear()


class LayoutAwareChunker:
    """Packs structural blocks into chunks while respecting document layout."""

    name = "layout_aware"

    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> list[ChunkDraft]:
        """Split ``document`` into layout-respecting chunks.

        Args:
            document: Parser output, consumed in reading order.
            config: Size and overlap tunables.

        Returns:
            Chunks with contiguous ordinals from zero. Empty or whitespace-only
            input yields an empty list, never a single empty chunk.

        Example:
            >>> blocks = (
            ...     TextBlock(text="1 Intro", kind=ChunkKind.HEADING, level=1),
            ...     TextBlock(text="Some prose."),
            ...     TextBlock(text="| a | b |", kind=ChunkKind.TABLE),
            ... )
            >>> chunks = LayoutAwareChunker().chunk(ParsedDocument(blocks=blocks), ChunkingConfig())
            >>> [(c.kind.value, c.section_path) for c in chunks]
            [('prose', ('1 Intro',)), ('table', ('1 Intro',))]
        """
        drafts: list[ChunkDraft] = []
        buffer = _Accumulator()
        section: tuple[str, ...] = ()
        # Overlap is only carried between prose chunks inside the same section.
        overlap_source: str = ""
        overlap_section: tuple[str, ...] = ()

        def flush() -> None:
            """Emit the buffered blocks as one chunk, if there is anything to emit."""
            nonlocal overlap_source, overlap_section
            if buffer.is_empty:
                return
            body = normalise_whitespace(buffer.text())
            buffer_blocks = list(buffer.blocks)
            buffer.clear()
            if not body:
                return

            carry = overlap_source if overlap_section == section else ""
            drafts.append(
                self._make_draft(
                    body=body,
                    blocks=buffer_blocks,
                    ordinal=len(drafts),
                    section=section,
                    config=config,
                    overlap_source=carry,
                )
            )
            overlap_source, overlap_section = body, section

        for block in document.blocks:
            if not block.text.strip():
                continue

            if block.kind is ChunkKind.HEADING:
                flush()
                section = self._advance_section(section, block)
                # A heading is metadata, not body text: it becomes the breadcrumb.
                overlap_source, overlap_section = "", ()
                continue

            budget = self._body_budget(section, config, has_overlap=bool(overlap_source))

            if block.is_atomic:
                flush()
                drafts.extend(
                    self._emit_atomic(
                        block=block,
                        start_ordinal=len(drafts),
                        section=section,
                        config=config,
                    )
                )
                # Never bleed table or code text into the next prose chunk.
                overlap_source, overlap_section = "", ()
                continue

            if block.char_length > budget:
                # A single paragraph larger than the budget: flush what we have,
                # then hard-split the paragraph so the ceiling still holds.
                # The split itself takes no overlap — `flush` already carries the
                # previous body forward, and doing both would overlap twice and
                # leave a mid-word fragment at each seam.
                flush()
                for piece in split_oversized_text(
                    block.text,
                    max_chars=self._body_budget(section, config, has_overlap=True),
                    overlap_chars=0,
                ):
                    buffer.blocks.append(block.model_copy(update={"text": piece}))
                    flush()
                continue

            if not buffer.is_empty and buffer.char_length + len(_JOIN) + block.char_length > budget:
                flush()

            buffer.blocks.append(block)

            if buffer.char_length >= config.target_chars:
                flush()

        flush()
        return self._merge_undersized(drafts, config)

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _breadcrumb(section: tuple[str, ...], config: ChunkingConfig) -> str:
        """Rendered heading prefix for a chunk, or an empty string."""
        if not (config.prepend_section_path and section):
            return ""
        return " > ".join(section) + _JOIN

    def _body_budget(
        self, section: tuple[str, ...], config: ChunkingConfig, *, has_overlap: bool
    ) -> int:
        """Characters available for body text once overhead is charged.

        Reserving the breadcrumb and overlap up front is what makes
        ``len(content) <= max_chars`` an invariant rather than an aspiration.
        Never returns less than a quarter of ``max_chars``, so a pathologically
        deep heading trail degrades the budget instead of collapsing it.
        """
        overhead = len(self._breadcrumb(section, config))
        if has_overlap:
            overhead += config.overlap_chars + 1
        return max(config.max_chars - overhead, config.max_chars // 4)

    @staticmethod
    def _advance_section(current: tuple[str, ...], heading: TextBlock) -> tuple[str, ...]:
        """Return the section breadcrumb after entering ``heading``.

        A level-2 heading replaces everything from depth 2 down, so
        ``("1 Intro", "1.1 Scope")`` followed by a level-2 "1.2 Aims" becomes
        ``("1 Intro", "1.2 Aims")``.

        Example:
            >>> LayoutAwareChunker._advance_section(
            ...     ("1 Intro", "1.1 Scope"), TextBlock(text="1.2 Aims", level=2)
            ... )
            ('1 Intro', '1.2 Aims')
        """
        if heading.section_path:
            return heading.section_path
        level = heading.level or (len(current) + 1)
        depth = max(level - 1, 0)
        return (*current[:depth], heading.text.strip())

    def _emit_atomic(
        self,
        *,
        block: TextBlock,
        start_ordinal: int,
        section: tuple[str, ...],
        config: ChunkingConfig,
    ) -> list[ChunkDraft]:
        """Emit a table or code block, splitting only if it is truly enormous."""
        if block.char_length <= config.max_chars * _ATOMIC_OVERSIZE_FACTOR:
            return [
                self._make_draft(
                    body=block.text,
                    blocks=[block],
                    ordinal=start_ordinal,
                    section=section,
                    config=config,
                    overlap_source="",
                )
            ]

        pieces = split_oversized_text(block.text, max_chars=config.max_chars, overlap_chars=0)
        return [
            self._make_draft(
                body=piece,
                blocks=[block],
                ordinal=start_ordinal + offset,
                section=section,
                config=config,
                overlap_source="",
                extra_metadata={"split_atomic": True, "part": offset + 1, "parts": len(pieces)},
            )
            for offset, piece in enumerate(pieces)
        ]

    def _make_draft(
        self,
        *,
        body: str,
        blocks: list[TextBlock],
        ordinal: int,
        section: tuple[str, ...],
        config: ChunkingConfig,
        overlap_source: str,
        extra_metadata: dict[str, object] | None = None,
    ) -> ChunkDraft:
        """Build one chunk from buffered blocks, applying overlap and breadcrumb."""
        content = body
        if config.overlap_chars and overlap_source:
            content = apply_overlap(overlap_source, content, overlap_chars=config.overlap_chars)
        content = self._breadcrumb(section, config) + content

        pages = [b.page_number for b in blocks if b.page_number is not None]
        return ChunkDraft(
            content=content,
            ordinal=ordinal,
            kind=self._dominant_kind(blocks),
            page_number=min(pages) if pages else None,
            section_path=section,
            bbox=self._union_bbox(blocks),
            metadata={
                "strategy": self.name,
                "block_count": len(blocks),
                **(extra_metadata or {}),
            },
        )

    @staticmethod
    def _dominant_kind(blocks: list[TextBlock]) -> ChunkKind:
        """Pick the chunk kind: an atomic block wins, otherwise the first non-heading."""
        for block in blocks:
            if block.is_atomic:
                return block.kind
        for block in blocks:
            if block.kind is not ChunkKind.HEADING:
                return block.kind
        return ChunkKind.PROSE

    @staticmethod
    def _union_bbox(blocks: list[TextBlock]) -> BoundingBox | None:
        """Smallest box containing every block's box, or None if none have one.

        Only meaningful when the blocks share a page; the draft keeps the minimum
        page number, so the box is a hint for the viewer, not a promise.
        """
        boxes = [b.bbox for b in blocks if b.bbox is not None]
        if not boxes:
            return None
        return BoundingBox(
            x0=min(b.x0 for b in boxes),
            y0=min(b.y0 for b in boxes),
            x1=max(b.x1 for b in boxes),
            y1=max(b.y1 for b in boxes),
        )

    @staticmethod
    def _merge_undersized(drafts: list[ChunkDraft], config: ChunkingConfig) -> list[ChunkDraft]:
        """Fold sub-``min_chars`` chunks into the previous chunk where it fits.

        A 40-character orphan retrieves badly and pollutes the citation list, so
        it is absorbed by its neighbour unless that would breach ``max_chars``,
        cross a section boundary, or merge an atomic block into prose.
        """
        if config.min_chars <= 0:
            return drafts

        merged: list[ChunkDraft] = []
        for draft in drafts:
            previous = merged[-1] if merged else None
            can_merge = (
                previous is not None
                and draft.char_length < config.min_chars
                and draft.kind is ChunkKind.PROSE
                and previous.kind is ChunkKind.PROSE
                and previous.section_path == draft.section_path
                and previous.char_length + len(_JOIN) + draft.char_length <= config.max_chars
            )
            if can_merge and previous is not None:
                merged[-1] = previous.model_copy(
                    update={"content": f"{previous.content}{_JOIN}{draft.content}"}
                )
                continue
            merged.append(draft)

        # Ordinals must stay contiguous after merging.
        return [d.model_copy(update={"ordinal": i}) for i, d in enumerate(merged)]


layout_aware_chunker = register_chunker(LayoutAwareChunker())
