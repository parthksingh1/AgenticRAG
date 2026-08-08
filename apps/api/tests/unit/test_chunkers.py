"""Chunker tests, example-based and property-based.

Chunking is where silent data loss hides: a chunker that drops the last
paragraph of every document produces a system that is subtly, permanently wrong
and passes every smoke test. The properties below are the invariants that make
that class of bug impossible to ship.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from src.ingestion.chunkers import available_chunkers, get_chunker
from src.ingestion.chunkers.base import (
    apply_overlap,
    normalise_whitespace,
    split_oversized_text,
    split_sentences,
)
from src.ingestion.chunkers.fixed import FixedSizeChunker
from src.ingestion.chunkers.late import LateChunker
from src.ingestion.chunkers.layout import LayoutAwareChunker
from src.ingestion.chunkers.semantic import SemanticChunker, cosine_distance, percentile
from src.ingestion.types import ChunkingConfig, ParsedDocument, TextBlock
from src.models.document import ChunkKind

pytestmark = pytest.mark.unit


# ── Hypothesis strategies ────────────────────────────────────────────────────

#: Printable text with real word boundaries; blank-only strings are allowed
#: because empty blocks are a real thing parsers emit.
prose_text = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=0,
    max_size=400,
)

block_strategy = st.builds(
    TextBlock,
    text=prose_text,
    kind=st.sampled_from([ChunkKind.PROSE, ChunkKind.HEADING, ChunkKind.TABLE, ChunkKind.CODE]),
    page_number=st.one_of(st.none(), st.integers(min_value=1, max_value=50)),
    level=st.one_of(st.none(), st.integers(min_value=1, max_value=4)),
)

document_strategy = st.builds(
    ParsedDocument,
    blocks=st.lists(block_strategy, min_size=0, max_size=12).map(tuple),
)


@st.composite
def config_strategy(draw: st.DrawFn) -> ChunkingConfig:
    """Generate valid, internally consistent chunking configs."""
    target = draw(st.integers(min_value=100, max_value=2000))
    max_chars = draw(st.integers(min_value=target, max_value=target * 3))
    return ChunkingConfig(
        target_chars=target,
        max_chars=max_chars,
        min_chars=draw(st.integers(min_value=0, max_value=target)),
        overlap_chars=draw(st.integers(min_value=0, max_value=target - 1)),
        prepend_section_path=draw(st.booleans()),
    )


HYPOTHESIS_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ── Properties that must hold for every layout-aware chunking ────────────────


@HYPOTHESIS_SETTINGS
@given(document=document_strategy, config=config_strategy())
def test_ordinals_are_contiguous_from_zero(
    document: ParsedDocument, config: ChunkingConfig
) -> None:
    """Ordinals index the chunk list exactly, with no gaps or duplicates.

    Citation markers are resolved by ordinal, so a gap here surfaces as a
    citation pointing at the wrong passage.
    """
    chunks = LayoutAwareChunker().chunk(document, config)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


@HYPOTHESIS_SETTINGS
@given(document=document_strategy, config=config_strategy())
def test_no_empty_chunks(document: ParsedDocument, config: ChunkingConfig) -> None:
    """An empty chunk is a wasted embedding and a broken citation target."""
    chunks = LayoutAwareChunker().chunk(document, config)
    assert all(chunk.content.strip() for chunk in chunks)


@HYPOTHESIS_SETTINGS
@given(document=document_strategy, config=config_strategy())
def test_prose_chunks_respect_the_size_ceiling(
    document: ParsedDocument, config: ChunkingConfig
) -> None:
    """Prose never exceeds ``max_chars`` once breadcrumb and overlap are charged.

    Atomic blocks are the documented exception, so they are excluded here and
    covered by their own test.
    """
    chunks = LayoutAwareChunker().chunk(document, config)
    oversized = [
        c for c in chunks if c.kind is ChunkKind.PROSE and c.char_length > config.max_chars
    ]
    assert not oversized, f"{len(oversized)} prose chunks exceeded max_chars={config.max_chars}"


@HYPOTHESIS_SETTINGS
@given(document=document_strategy, config=config_strategy())
def test_no_source_text_is_lost(document: ParsedDocument, config: ChunkingConfig) -> None:
    """Every non-heading source block survives into at least one chunk.

    This is the property that catches the "dropped the last paragraph" bug. It
    compares on collapsed whitespace because chunkers are allowed to normalise.
    """
    chunks = LayoutAwareChunker().chunk(document, config)
    haystack = " ".join(" ".join(c.content.split()) for c in chunks)

    for block in document.blocks:
        if block.kind is ChunkKind.HEADING or not block.text.strip():
            continue
        needle = " ".join(block.text.split())
        # Long blocks get hard-split, so check a distinctive prefix instead.
        probe = needle[: min(len(needle), 40)]
        assert probe in haystack, f"block text vanished from every chunk: {probe!r}"


@HYPOTHESIS_SETTINGS
@given(document=document_strategy, config=config_strategy())
def test_chunking_is_deterministic(document: ParsedDocument, config: ChunkingConfig) -> None:
    """Same input, same output — required for retries and for snapshot evals."""
    chunker = LayoutAwareChunker()
    first = chunker.chunk(document, config)
    second = chunker.chunk(document, config)
    assert [c.content for c in first] == [c.content for c in second]


@HYPOTHESIS_SETTINGS
@given(document=document_strategy, config=config_strategy())
def test_tables_are_never_split_across_chunks(
    document: ParsedDocument, config: ChunkingConfig
) -> None:
    """A table block appears whole in exactly one chunk unless it is enormous."""
    chunks = LayoutAwareChunker().chunk(document, config)
    for block in document.blocks:
        if block.kind is not ChunkKind.TABLE or not block.text.strip():
            continue
        if block.char_length > config.max_chars * 2:
            continue  # documented oversize path, flagged in metadata
        # Degenerate single-character text can also appear inside a prose
        # chunk, so the property is "some TABLE chunk holds it whole", not
        # "the first chunk containing this substring is a table".
        table_chunks = [c for c in chunks if c.kind is ChunkKind.TABLE and block.text in c.content]
        assert table_chunks, f"table block was not preserved whole: {block.text[:40]!r}"


# ── Cross-strategy invariants ────────────────────────────────────────────────


@pytest.mark.parametrize("strategy", ["layout_aware", "fixed"])
def test_registered_strategies_handle_an_empty_document(strategy: str) -> None:
    """An empty document yields no chunks rather than one empty chunk."""
    chunker = get_chunker(strategy)
    assert chunker.chunk(ParsedDocument(blocks=()), ChunkingConfig()) == []


def test_registry_lists_the_dependency_free_strategies() -> None:
    """``semantic`` and ``late`` stay unregistered until a model is bound."""
    assert set(available_chunkers()) == {"layout_aware", "fixed"}


def test_unknown_strategy_raises_with_a_helpful_message() -> None:
    """A typo in tenant config fails loudly, listing what is available."""
    with pytest.raises(KeyError, match="registered: fixed, layout_aware"):
        get_chunker("layout-aware")


# ── Layout-aware specifics ───────────────────────────────────────────────────


def test_heading_becomes_a_breadcrumb_not_body_text() -> None:
    """The heading is carried as a prefix exactly once, not duplicated into the body."""
    doc = ParsedDocument(
        blocks=(
            TextBlock(text="1 Introduction", kind=ChunkKind.HEADING, level=1),
            TextBlock(text="Retrieval augments generation."),
        )
    )
    chunks = LayoutAwareChunker().chunk(doc, ChunkingConfig())

    assert len(chunks) == 1
    assert chunks[0].content == "1 Introduction\n\nRetrieval augments generation."
    assert chunks[0].section_path == ("1 Introduction",)


def test_sections_are_not_mixed_within_a_chunk() -> None:
    """Content from two sections never lands in the same chunk."""
    doc = ParsedDocument(
        blocks=(
            TextBlock(text="1 Intro", kind=ChunkKind.HEADING, level=1),
            TextBlock(text="Alpha."),
            TextBlock(text="2 Method", kind=ChunkKind.HEADING, level=1),
            TextBlock(text="Beta."),
        )
    )
    chunks = LayoutAwareChunker().chunk(doc, ChunkingConfig(min_chars=0))

    assert len(chunks) == 2
    assert "Beta." not in chunks[0].content
    assert "Alpha." not in chunks[1].content


def test_nested_heading_replaces_only_its_own_depth() -> None:
    """A level-2 heading keeps the level-1 breadcrumb above it."""
    doc = ParsedDocument(
        blocks=(
            TextBlock(text="1 Intro", kind=ChunkKind.HEADING, level=1),
            TextBlock(text="Alpha."),
            TextBlock(text="1.1 Scope", kind=ChunkKind.HEADING, level=2),
            TextBlock(text="Beta."),
        )
    )
    chunks = LayoutAwareChunker().chunk(doc, ChunkingConfig(min_chars=0))

    assert chunks[-1].section_path == ("1 Intro", "1.1 Scope")


def test_table_text_does_not_bleed_into_the_next_prose_chunk() -> None:
    """Overlap is suppressed across an atomic block."""
    doc = ParsedDocument(
        blocks=(
            TextBlock(text="| col | val |", kind=ChunkKind.TABLE),
            TextBlock(text="The table above shows revenue."),
        )
    )
    chunks = LayoutAwareChunker().chunk(doc, ChunkingConfig(overlap_chars=100, min_chars=0))

    prose = [c for c in chunks if c.kind is ChunkKind.PROSE]
    assert prose
    assert all("| col |" not in c.content for c in prose)


def test_overlap_starts_on_a_word_boundary() -> None:
    """A chunk never begins with half a word, which reads as corruption in the UI."""
    long_paragraph = "Retrieval augmented generation improves factual accuracy. " * 30
    doc = ParsedDocument(blocks=(TextBlock(text=long_paragraph),))
    chunks = LayoutAwareChunker().chunk(
        doc, ChunkingConfig(target_chars=400, max_chars=600, overlap_chars=80)
    )

    assert len(chunks) > 1
    for chunk in chunks[1:]:
        first_word = chunk.content.split()[0].strip(".,")
        assert first_word in long_paragraph.split(), f"chunk begins mid-word: {first_word!r}"


def test_undersized_chunks_are_merged_into_their_neighbour() -> None:
    """Orphan fragments are absorbed rather than emitted as their own chunk."""
    doc = ParsedDocument(
        blocks=(TextBlock(text="A full sentence of reasonable length here."), TextBlock(text="Hi."))
    )
    chunks = LayoutAwareChunker().chunk(doc, ChunkingConfig(min_chars=100))

    assert len(chunks) == 1
    assert "Hi." in chunks[0].content


# ── Fixed baseline ───────────────────────────────────────────────────────────


def test_fixed_chunker_ignores_structure_by_design() -> None:
    """The baseline flattens headings and tables into undifferentiated prose."""
    doc = ParsedDocument(
        blocks=(
            TextBlock(text="1 Intro", kind=ChunkKind.HEADING, level=1),
            TextBlock(text="| a | b |", kind=ChunkKind.TABLE),
        )
    )
    chunks = FixedSizeChunker().chunk(doc, ChunkingConfig())

    assert all(c.kind is ChunkKind.PROSE for c in chunks)
    assert all(c.section_path == () for c in chunks)


# ── Semantic chunker ─────────────────────────────────────────────────────────


def _topic_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic fake embedder: two orthogonal topics keyed by a marker word."""
    return [[0.0, 1.0] if "finance" in t.lower() else [1.0, 0.0] for t in texts]


def test_semantic_chunker_cuts_at_the_topic_change() -> None:
    """A sharp topic shift becomes a chunk boundary."""
    text = (
        "Neural retrieval encodes text into vectors. "
        "Dense vectors capture semantics well. "
        "The finance team reported quarterly revenue. "
        "Our finance results exceeded the forecast."
    )
    doc = ParsedDocument(blocks=(TextBlock(text=text),))
    chunker = SemanticChunker(embed=_topic_embed, breakpoint_percentile=75.0)

    chunks = chunker.chunk(doc, ChunkingConfig(target_chars=800, max_chars=1000))

    assert len(chunks) == 2
    assert "finance" not in chunks[0].content.lower()
    assert "finance" in chunks[1].content.lower()


def test_semantic_chunker_rejects_an_inconsistent_embedder() -> None:
    """An embedder returning the wrong number of vectors fails loudly."""
    doc = ParsedDocument(blocks=(TextBlock(text="One. Two. Three."),))
    chunker = SemanticChunker(embed=lambda _texts: [[1.0]])

    with pytest.raises(ValueError, match="different number of vectors"):
        chunker.chunk(doc, ChunkingConfig())


@pytest.mark.parametrize("bad", [10.0, 99.95])
def test_semantic_chunker_validates_its_percentile(bad: float) -> None:
    """A nonsensical breakpoint percentile is rejected at construction."""
    with pytest.raises(ValueError, match="breakpoint_percentile"):
        SemanticChunker(embed=_topic_embed, breakpoint_percentile=bad)


# ── Late chunker ─────────────────────────────────────────────────────────────


def test_late_chunker_attaches_a_context_aware_vector() -> None:
    """Chunks carry a pooled vector derived from the whole-document encoding."""
    doc = ParsedDocument(
        blocks=(TextBlock(text="Alpha beta gamma."), TextBlock(text="Delta epsilon zeta."))
    )
    chunker = LateChunker(encode=lambda text: [[float(i), 1.0] for i in range(len(text))])

    chunks = chunker.chunk(
        doc, ChunkingConfig(target_chars=200, max_chars=240, min_chars=0, overlap_chars=0)
    )

    assert chunks
    assert any("late_embedding" in c.metadata for c in chunks)
    for chunk in chunks:
        vector = chunk.metadata.get("late_embedding")
        if vector:
            assert len(vector) == 2


def test_late_chunker_rejects_a_short_encoder_output() -> None:
    """An encoder returning fewer vectors than characters is a contract violation."""
    doc = ParsedDocument(blocks=(TextBlock(text="Alpha beta gamma delta."),))
    chunker = LateChunker(encode=lambda _text: [[1.0]])

    with pytest.raises(ValueError, match="at least one vector per character"):
        chunker.chunk(doc, ChunkingConfig())


def test_late_chunker_requires_a_usable_context_window() -> None:
    """A context window too small to be worth the technique is rejected."""
    with pytest.raises(ValueError, match="max_context_chars"):
        LateChunker(encode=lambda text: [[1.0]] * len(text), max_context_chars=10)


# ── Shared helpers ───────────────────────────────────────────────────────────


@given(
    text=st.text(min_size=0, max_size=2000),
    max_chars=st.integers(min_value=10, max_value=200),
)
@settings(max_examples=200, deadline=None)
def test_split_oversized_text_respects_its_ceiling(text: str, max_chars: int) -> None:
    """Every piece fits, and the split always terminates."""
    pieces = split_oversized_text(text, max_chars=max_chars)
    assert all(len(piece) <= max_chars for piece in pieces)


@given(text=st.text(min_size=1, max_size=2000), max_chars=st.integers(min_value=10, max_value=200))
@settings(max_examples=200, deadline=None)
def test_split_oversized_text_without_overlap_preserves_every_character(
    text: str, max_chars: int
) -> None:
    """With no overlap, the pieces concatenate back to the original exactly."""
    assert "".join(split_oversized_text(text, max_chars=max_chars, overlap_chars=0)) == text


def test_split_oversized_text_rejects_overlap_at_or_above_the_ceiling() -> None:
    """An overlap that cannot advance would loop forever, so it is rejected."""
    with pytest.raises(ValueError, match="smaller than max_chars"):
        split_oversized_text("abcdef", max_chars=4, overlap_chars=4)


@pytest.mark.parametrize(
    ("previous", "current", "overlap", "expected"),
    [
        ("the quick brown fox", "jumps over", 9, "brown fox jumps over"),
        ("the quick brown fox", "jumps over", 7, "fox jumps over"),
        ("anything", "current", 0, "current"),
        ("", "current", 20, "current"),
    ],
)
def test_apply_overlap_trims_only_partial_words(
    previous: str, current: str, overlap: int, expected: str
) -> None:
    """A cut on a word boundary keeps the full overlap; a mid-word cut is trimmed."""
    assert apply_overlap(previous, current, overlap_chars=overlap) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two! Three?", ["One.", "Two!", "Three?"]),
        ("See Fig. 2 for detail. It shows growth.", ["See Fig. 2 for detail.", "It shows growth."]),
        ("", []),
        ("   ", []),
        ("No terminator", ["No terminator"]),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    """Sentence splitting keeps terminators and survives common abbreviations."""
    assert split_sentences(text) == expected


def test_normalise_whitespace_preserves_paragraphs() -> None:
    """Blank-line runs collapse to a single paragraph break, not to nothing."""
    assert normalise_whitespace("a  \n\n\n\nb   ") == "a\n\nb"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [([1.0, 0.0], [1.0, 0.0], 0.0), ([1.0, 0.0], [0.0, 1.0], 1.0), ([0.0, 0.0], [1.0, 0.0], 1.0)],
)
def test_cosine_distance(a: list[float], b: list[float], expected: float) -> None:
    """Cosine distance, with zero vectors treated as maximally distant."""
    assert cosine_distance(a, b) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [([1.0, 2.0, 3.0, 4.0], 50, 2.5), ([5.0], 95, 5.0), ([1.0, 2.0], 100, 2.0)],
)
def test_percentile(values: list[float], q: float, expected: float) -> None:
    """Linear-interpolated percentile matches the numpy convention."""
    assert percentile(values, q) == pytest.approx(expected)


def test_percentile_of_empty_sequence_raises() -> None:
    """An empty distance distribution is a bug, not a zero."""
    with pytest.raises(ValueError, match="empty sequence"):
        percentile([], 50)
