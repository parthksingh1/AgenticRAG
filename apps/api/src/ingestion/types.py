"""Typed contracts shared by parsers, chunkers and embedders.

The pipeline is deliberately three pure stages over immutable data::

    bytes ──parser──▶ ParsedDocument ──chunker──▶ [ChunkDraft] ──embedder──▶ [EmbeddedChunk]

Only the service layer touches the database, so every stage here is a pure
function that can be property-tested without a container in sight.

Example:
    >>> block = TextBlock(text="Hello world", kind=ChunkKind.PROSE, page_number=1)
    >>> block.char_length
    11
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.document import ChunkKind


class BoundingBox(BaseModel):
    """Page-space rectangle, normalised to 0-1 so it survives rescaling.

    The document viewer uses this to highlight the exact region a citation came
    from, which is why it is carried all the way through to the chunk row.
    """

    model_config = ConfigDict(frozen=True)

    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> Self:
        """Reject inverted rectangles rather than silently normalising them."""
        if self.x1 < self.x0 or self.y1 < self.y0:
            msg = "bounding box corners are inverted"
            raise ValueError(msg)
        return self


class TextBlock(BaseModel):
    """One structural element recovered by a parser.

    A block is the smallest thing a parser is willing to vouch for: a paragraph,
    a table, a code listing, a heading. Chunkers group and split blocks but never
    invent them, which is what lets the layout-aware chunker promise that a table
    is never cut in half.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    kind: ChunkKind = ChunkKind.PROSE
    page_number: int | None = None
    #: Enclosing heading trail, outermost first, e.g. ``["2 Method", "2.1 Setup"]``.
    section_path: tuple[str, ...] = ()
    bbox: BoundingBox | None = None
    #: Heading depth, 1-6. Only meaningful when ``kind`` is HEADING.
    level: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def char_length(self) -> int:
        """Number of characters in the block's text."""
        return len(self.text)

    @property
    def is_atomic(self) -> bool:
        """True when the block must never be split across chunks.

        Splitting a table or a code listing destroys the very structure that
        makes it retrievable, so these are kept whole even when they exceed the
        target chunk size.
        """
        return self.kind in (ChunkKind.TABLE, ChunkKind.CODE)

    def section_label(self) -> str:
        """Render the section path as a breadcrumb string.

        Example:
            >>> TextBlock(text="x", section_path=("2 Method", "2.1 Setup")).section_label()
            '2 Method > 2.1 Setup'
        """
        return " > ".join(self.section_path)


class ParsedDocument(BaseModel):
    """A document after parsing, before chunking."""

    model_config = ConfigDict(frozen=True)

    blocks: tuple[TextBlock, ...]
    title: str | None = None
    parser: str = "unknown"
    page_count: int | None = None
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """The document's text with blocks joined by blank lines."""
        return "\n\n".join(block.text for block in self.blocks)

    @property
    def char_length(self) -> int:
        """Total characters across all blocks (excluding join separators)."""
        return sum(block.char_length for block in self.blocks)


class ChunkDraft(BaseModel):
    """A chunk produced by a chunker, before embedding and persistence.

    ``content`` is what a user will see as a citation. ``embedded_text`` is what
    actually gets vectorised — for Contextual Retrieval that includes a generated
    preamble, so the two deliberately differ.
    """

    model_config = ConfigDict(frozen=True)

    content: str
    ordinal: int = Field(ge=0)
    kind: ChunkKind = ChunkKind.PROSE
    page_number: int | None = None
    section_path: tuple[str, ...] = ()
    bbox: BoundingBox | None = None
    #: Character offsets into ``ParsedDocument.full_text``. Used by the version
    #: diff view and to prove chunk coverage in the property tests.
    char_start: int | None = None
    char_end: int | None = None
    context_preamble: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def embedded_text(self) -> str:
        r"""The text that should be sent to the embedding model.

        Example:
            >>> ChunkDraft(content="Revenue grew 12%.", ordinal=0).embedded_text
            'Revenue grew 12%.'
            >>> ChunkDraft(
            ...     content="Revenue grew 12%.", ordinal=0, context_preamble="Q3 2024 results."
            ... ).embedded_text
            'Q3 2024 results.\n\nRevenue grew 12%.'
        """
        if self.context_preamble:
            return f"{self.context_preamble}\n\n{self.content}"
        return self.content

    @property
    def char_length(self) -> int:
        """Length of the visible chunk content."""
        return len(self.content)


class EmbeddedChunk(BaseModel):
    """A chunk with its dense vector attached."""

    model_config = ConfigDict(frozen=True)

    draft: ChunkDraft
    embedding: tuple[float, ...]
    model: str
    token_count: int = 0

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vector."""
        return len(self.embedding)


class ChunkingConfig(BaseModel):
    """Tunables shared by every chunking strategy.

    Sizes are in characters rather than tokens because chunkers run before
    tokenisation and must behave identically regardless of which tenant's
    embedding model is configured. The token budget is enforced downstream by
    the embedder, which truncates and warns.
    """

    model_config = ConfigDict(frozen=True)

    target_chars: int = Field(default=1200, ge=100, le=8000)
    #: Defaults to 5/3 of ``target_chars`` when not given explicitly.
    max_chars: int = Field(default=2000, ge=100, le=16000)
    #: Chunks below this are merged into a neighbour rather than emitted alone.
    #: Defaults to a sixth of ``target_chars``.
    min_chars: int = Field(default=200, ge=0)
    #: Defaults to an eighth of ``target_chars``.
    overlap_chars: int = Field(default=150, ge=0)
    #: Keep the heading breadcrumb at the top of each chunk. Cheap, and measurably
    #: improves retrieval on documents with repetitive section wording.
    prepend_section_path: bool = True

    @model_validator(mode="before")
    @classmethod
    def _scale_derived_sizes(cls, data: Any) -> Any:
        """Scale unset size fields to the caller's ``target_chars``.

        Without this, ``ChunkingConfig(target_chars=300)`` would fail validation
        against the absolute defaults, which is a trap for anyone tuning the
        chunker. Explicitly supplied values are never overridden.

        Example:
            >>> c = ChunkingConfig(target_chars=400)
            >>> (c.max_chars, c.min_chars, c.overlap_chars)
            (666, 66, 50)
            >>> ChunkingConfig(target_chars=400, overlap_chars=10).overlap_chars
            10
        """
        if not isinstance(data, dict) or "target_chars" not in data:
            return data
        target = data["target_chars"]
        if not isinstance(target, int):
            return data
        derived = {
            "max_chars": target * 5 // 3,
            "min_chars": target // 6,
            "overlap_chars": target // 8,
        }
        return {**derived, **data}

    @model_validator(mode="after")
    def _check_sizes(self) -> Self:
        """Reject configurations whose sizes cannot all be satisfied at once."""
        if self.max_chars < self.target_chars:
            msg = "max_chars must be >= target_chars"
            raise ValueError(msg)
        if self.overlap_chars >= self.target_chars:
            msg = "overlap_chars must be < target_chars, otherwise chunking cannot advance"
            raise ValueError(msg)
        if self.min_chars > self.target_chars:
            msg = "min_chars must be <= target_chars"
            raise ValueError(msg)
        return self
