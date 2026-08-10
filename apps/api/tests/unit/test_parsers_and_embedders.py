"""Parser and embedder tests.

Parsing is where silent corruption enters a RAG system: a PDF misread as text
produces confident nonsense that is then indexed, retrieved and cited. These
tests pin the detection and structure-recovery behaviour that prevents that.
"""

from __future__ import annotations

import pytest
from src.core.errors import UnsupportedMediaTypeError
from src.ingestion.embedders.base import (
    CachingEmbedder,
    HashingEmbedder,
    InMemoryEmbeddingCache,
    deduplicate,
    dynamic_batch_size,
    truncate,
)
from src.ingestion.embedders.contextual import ContextualEnricher, _prepare_document
from src.ingestion.parsers import detect_format, get_parser, parse_bytes, supported_formats
from src.ingestion.parsers.text import CsvParser, MarkdownParser, TextParser, decode
from src.ingestion.parsers.web import HtmlParser, _strip_tags
from src.ingestion.types import ChunkDraft
from src.models.document import ChunkKind
from src.services.llm import pricing
from src.services.llm.providers import FakeProvider
from src.services.llm.router import LLMRouter, ModelPolicy

pytestmark = pytest.mark.unit


# ── Format detection ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("data", "filename", "expected"),
    [
        (b"%PDF-1.7\n...", None, "pdf"),
        (b"\x89PNG\r\n\x1a\n", None, "image"),
        (b"\xff\xd8\xff\xe0", "photo.bin", "image"),
        (b"<!DOCTYPE html><html></html>", None, "html"),
        (b"<html><body>x</body></html>", None, "html"),
        (b"a,b\n1,2\n", "data.csv", "csv"),
        (b"# Title", "notes.md", "markdown"),
        (b"plain", "log.txt", "text"),
        (b"plain", None, "text"),
    ],
)
def test_format_detection(data: bytes, filename: str | None, expected: str) -> None:
    """Detection covers magic bytes, markup sniffing and extensions, in that order."""
    assert detect_format(data, filename=filename) == expected


def test_content_beats_a_lying_filename() -> None:
    """A PDF named .txt must be parsed as a PDF, not indexed as binary noise.

    This is the highest-value case in the whole detection path: parsing a PDF as
    text succeeds, produces garbage, and nothing downstream ever notices.
    """
    assert detect_format(b"%PDF-1.4 binary", filename="totally_a.txt") == "pdf"


def test_ooxml_containers_are_disambiguated_by_their_contents() -> None:
    """DOCX, PPTX and XLSX are all ZIPs; only the entry names tell them apart."""
    assert detect_format(b"PK\x03\x04" + b"\x00" * 20 + b"word/document.xml") == "docx"
    assert detect_format(b"PK\x03\x04" + b"\x00" * 20 + b"ppt/slides/slide1.xml") == "pptx"
    assert detect_format(b"PK\x03\x04" + b"\x00" * 20 + b"xl/workbook.xml") == "xlsx"


def test_every_advertised_format_has_a_parser() -> None:
    """The registry and the detector must not drift apart."""
    for fmt in supported_formats():
        assert get_parser(fmt) is not None


def test_unknown_format_raises_a_415_not_a_crash() -> None:
    """The API answers 415 rather than failing deep inside a Celery task."""
    with pytest.raises(UnsupportedMediaTypeError) as excinfo:
        get_parser("application/x-nonsense")

    assert excinfo.value.status_code == 415


# ── Text and Markdown ────────────────────────────────────────────────────────


def test_text_parser_splits_on_blank_lines() -> None:
    """Paragraph structure is the only structure plain text has."""
    doc = TextParser().parse(b"First para.\n\nSecond para.\n\n\nThird.")

    assert [b.text for b in doc.blocks] == ["First para.", "Second para.", "Third."]


def test_markdown_recovers_headings_code_tables_and_lists() -> None:
    """Markdown gets the same block structure a PDF does, so chunking is uniform."""
    source = b"""# Guide

Intro prose.

## Setup

- one
- two

```python
x = 1
```

| a | b |
| - | - |
| 1 | 2 |
"""
    doc = MarkdownParser().parse(source)
    kinds = [b.kind.value for b in doc.blocks]

    assert kinds == ["heading", "prose", "heading", "list", "code", "table"]
    assert doc.title == "Guide"


def test_markdown_nested_headings_build_a_section_path() -> None:
    """The breadcrumb is what lets the chunker keep sections apart."""
    doc = MarkdownParser().parse(b"# A\n\n## B\n\ntext\n")

    body = next(b for b in doc.blocks if b.kind is ChunkKind.PROSE)
    assert body.section_path == ("A", "B")


def test_code_fences_are_not_split_or_reinterpreted() -> None:
    """A hash inside a code fence is a comment, not a heading."""
    doc = MarkdownParser().parse(b"```python\n# not a heading\nx = 1\n```\n")

    assert len(doc.blocks) == 1
    assert doc.blocks[0].kind is ChunkKind.CODE
    assert "# not a heading" in doc.blocks[0].text


@pytest.mark.parametrize(
    ("data", "expected"),
    [(b"hello", "hello"), ("café".encode("latin-1"), "café"), ("naïve".encode(), "naïve")],
)
def test_decode_tries_the_encodings_uploads_actually_use(data: bytes, expected: str) -> None:
    """Refusing a mis-encoded file is a frustrating failure a user cannot fix."""
    assert decode(data) == expected


def test_decode_never_raises_on_undecodable_bytes() -> None:
    """A few bad bytes must not cost the whole document."""
    assert decode(b"\xff\xfe\x00valid text") != ""


# ── CSV ──────────────────────────────────────────────────────────────────────


def test_csv_becomes_an_atomic_table_block() -> None:
    """Tabular data must stay whole so the chunker does not shred it."""
    doc = CsvParser().parse(b"name,amount\nalice,10\nbob,20\n")

    tables = [b for b in doc.blocks if b.kind is ChunkKind.TABLE]
    assert len(tables) == 1
    assert "| name | amount |" in tables[0].text
    assert "| alice | 10 |" in tables[0].text


def test_csv_records_its_schema_for_the_sql_mcp_server() -> None:
    """Column names in metadata are what makes the analytics tool usable."""
    doc = CsvParser().parse(b"region,revenue\nemea,100\n")

    assert doc.metadata["columns"] == ["region", "revenue"]
    assert doc.metadata["row_count"] == 1


def test_csv_detects_a_semicolon_delimiter() -> None:
    """European CSV exports are semicolon-delimited and are not an edge case."""
    doc = CsvParser().parse(b"name;amount\nalice;10\n")

    assert doc.metadata["columns"] == ["name", "amount"]


def test_ragged_rows_do_not_break_the_table() -> None:
    """Real CSVs have short rows; the renderer pads rather than misaligning."""
    doc = CsvParser().parse(b"a,b,c\n1,2\n")

    table = next(b for b in doc.blocks if b.kind is ChunkKind.TABLE)
    assert table.text.count("|") % 2 == 0


def test_a_cell_containing_a_pipe_cannot_break_the_table() -> None:
    """Otherwise one cell silently corrupts every row below it."""
    doc = CsvParser().parse(b'a,b\n"x|y",2\n')

    table = next(b for b in doc.blocks if b.kind is ChunkKind.TABLE)
    assert "x\\|y" in table.text


def test_empty_csv_yields_no_blocks() -> None:
    """An empty upload is not an error, it is just empty."""
    assert CsvParser().parse(b"").blocks == ()


# ── HTML ─────────────────────────────────────────────────────────────────────


def test_html_extraction_drops_scripts_and_styles() -> None:
    """Script bodies retrieved as content would match everything and answer nothing."""
    doc = HtmlParser().parse(
        b"<html><body><p>Real content.</p>"
        b"<script>var tracking = 1;</script>"
        b"<style>.a{color:red}</style></body></html>"
    )
    text = " ".join(b.text for b in doc.blocks)

    assert "Real content." in text
    assert "tracking" not in text
    assert "color:red" not in text


def test_html_title_is_extracted_and_unescaped() -> None:
    """Entities in the title would otherwise show up literally in the UI."""
    doc = HtmlParser().parse(b"<html><head><title>A &amp; B</title></head><body>x</body></html>")

    assert doc.title == "A & B"


def test_strip_tags_splits_on_block_boundaries() -> None:
    """Without block splitting the whole page becomes one unusable chunk."""
    assert _strip_tags("<p>One</p><p>Two</p>") == ["One", "Two"]


# ── Embedders ────────────────────────────────────────────────────────────────


async def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    """Offline determinism is what makes the ingestion tests and demo reproducible."""
    embedder = HashingEmbedder(dimension=64)

    first = await embedder.embed(["retrieval augmented generation"])
    second = await embedder.embed(["retrieval augmented generation"])

    assert first == second
    magnitude = sum(v * v for v in first[0]) ** 0.5
    assert magnitude == pytest.approx(1.0)


async def test_identical_texts_share_a_vector() -> None:
    """Deduplication must not change results, only cost."""
    vectors = await HashingEmbedder(dimension=32).embed(["same", "same", "different"])

    assert vectors[0] == vectors[1]
    assert vectors[0] != vectors[2]


def test_deduplicate_returns_a_faithful_index_map() -> None:
    """Rehydrating through the map must reproduce the original ordering."""
    unique, mapping = deduplicate(["a", "b", "a", "c", "b"])

    assert unique == ["a", "b", "c"]
    assert [unique[i] for i in mapping] == ["a", "b", "a", "c", "b"]


@pytest.mark.parametrize(
    ("texts", "expected"),
    [(["short"] * 100, 64), (["x" * 30_000] * 10, 2), ([], 1)],
)
def test_batch_size_adapts_to_text_length(texts: list[str], expected: int) -> None:
    """A fixed batch size is wasteful on short chunks and fatal on long ones."""
    assert dynamic_batch_size(texts) == expected


def test_oversized_text_is_truncated_rather_than_rejected() -> None:
    """A provider 400 mid-ingestion is worse than a truncated tail."""
    assert len(truncate("x" * 50_000, limit=100)) == 100


async def test_caching_embedder_serves_repeats_without_recomputing() -> None:
    """Re-ingesting unchanged text should cost nothing."""
    inner = HashingEmbedder(dimension=16)
    embedder = CachingEmbedder(inner, cache=InMemoryEmbeddingCache())

    first = await embedder.embed(["alpha", "beta"])
    second = await embedder.embed(["alpha", "beta"])

    assert first == second
    assert embedder.hits == 2
    assert embedder.misses == 2
    assert embedder.hit_ratio == 0.5


async def test_cache_keys_are_namespaced_by_model() -> None:
    """Serving a vector from the wrong embedding space fails silently, so it must not happen."""
    cache = InMemoryEmbeddingCache()
    a = CachingEmbedder(HashingEmbedder(dimension=8, model_name="model-a"), cache=cache)
    b = CachingEmbedder(HashingEmbedder(dimension=8, model_name="model-b"), cache=cache)

    await a.embed(["shared text"])
    await b.embed(["shared text"])

    assert b.hits == 0, "model-b must not read model-a's vectors"


async def test_partial_cache_hit_only_recomputes_the_misses() -> None:
    """The common case during re-ingestion: most chunks unchanged, a few new."""
    embedder = CachingEmbedder(HashingEmbedder(dimension=8), cache=InMemoryEmbeddingCache())

    await embedder.embed(["a"])
    await embedder.embed(["a", "b"])

    assert (embedder.hits, embedder.misses) == (1, 2)


# ── Contextual retrieval ─────────────────────────────────────────────────────


@pytest.fixture
def contextual_router() -> LLMRouter:
    """A router whose fake provider always returns a fixed context sentence."""
    pricing.MODEL_PROVIDERS["ctx-test-model"] = "fake"
    return LLMRouter(
        providers={"fake": FakeProvider(responses=["From the ACME 2024 annual report."])},
        policy=ModelPolicy(default_model="ctx-test-model"),
    )


async def test_context_is_embedded_but_never_displayed(contextual_router: LLMRouter) -> None:
    """Showing generated text as source material would be a quiet fabrication."""
    drafts = [ChunkDraft(content="Revenue grew 12%.", ordinal=0)]

    enriched = await ContextualEnricher(router=contextual_router).enrich(
        drafts, document_text="annual report body", document_title="ACME 2024"
    )

    assert enriched[0].content == "Revenue grew 12%."
    assert enriched[0].embedded_text.startswith("From the ACME 2024 annual report.")


async def test_enrichment_failure_degrades_instead_of_failing_ingestion() -> None:
    """Degraded retrieval beats a failed upload."""
    pricing.MODEL_PROVIDERS["ctx-dead-model"] = "fake"
    router = LLMRouter(
        providers={"fake": FakeProvider(fail_times=99)},
        policy=ModelPolicy(default_model="ctx-dead-model"),
    )
    enricher = ContextualEnricher(router=router)

    enriched = await enricher.enrich(
        [ChunkDraft(content="Revenue grew 12%.", ordinal=0)], document_text="doc"
    )

    assert enriched[0].context_preamble is None
    assert enriched[0].embedded_text == "Revenue grew 12%."
    assert enricher.failed == 1


async def test_enrichment_of_no_chunks_makes_no_calls(contextual_router: LLMRouter) -> None:
    """An empty document must not cost a provider round trip."""
    assert await ContextualEnricher(router=contextual_router).enrich([], document_text="") == []


def test_document_context_keeps_both_ends_of_a_long_document() -> None:
    """The head carries framing, the tail carries conclusions; a prefix loses one."""
    body = "HEAD" + ("x" * 100_000) + "TAIL"

    prepared = _prepare_document(body, title=None)

    assert prepared.startswith("HEAD")
    assert prepared.endswith("TAIL")
    assert "[...]" in prepared


# ── End-to-end through the registry ──────────────────────────────────────────


def test_parse_bytes_dispatches_without_the_caller_naming_a_format() -> None:
    """Ingestion calls one function; the registry does the rest."""
    doc = parse_bytes(b"# Heading\n\nBody text.", filename="notes.md")

    assert doc.parser == "markdown"
    assert doc.title == "Heading"
