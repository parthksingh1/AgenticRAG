"""PDF parsing with layout recovery.

PyMuPDF gives text with positions; turning that into *structure* is the work.
Two things are recovered that plain text extraction throws away:

* **Headings**, inferred from font size relative to the document's own body-text
  size. An absolute threshold fails immediately across documents with different
  base sizes, so the modal span size is treated as body text and anything
  meaningfully larger is a heading, ranked into levels.
* **Tables**, via PyMuPDF's table finder, emitted as atomic Markdown blocks with
  their bounding boxes so the viewer can highlight them.

Pages that yield almost no text are routed to OCR, because a scanned page is
indistinguishable from a blank one until you look at how much text came back.

Example:
    >>> PdfParser().name
    'pymupdf'
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.errors import IngestionError
from src.core.logging import get_logger
from src.ingestion.parsers.base import Parser, register_parser
from src.ingestion.types import BoundingBox, ParsedDocument, TextBlock
from src.models.document import ChunkKind

if TYPE_CHECKING:
    from collections.abc import Sequence

log = get_logger(__name__)

#: A page yielding fewer characters than this is treated as scanned and sent to
#: OCR. Tuned high enough to catch pages carrying only a header or page number.
OCR_TRIGGER_CHARS = 60

#: A span must be this much larger than body text to count as a heading.
HEADING_SIZE_RATIO = 1.15


class PdfParser(Parser):
    """Extracts structured blocks from a PDF."""

    name = "pymupdf"
    formats = frozenset({"pdf"})

    def __init__(self, *, ocr_fallback: bool = True) -> None:
        """Create the parser.

        Args:
            ocr_fallback: Whether to OCR pages that yield almost no text.
        """
        self._ocr_fallback = ocr_fallback

    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        """Parse a PDF into headings, prose and table blocks."""
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - dependency present in the image
            raise IngestionError("PyMuPDF is not installed; cannot parse PDF") from exc

        try:
            document = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise IngestionError(f"could not open PDF: {exc}") from exc

        try:
            spans = self._collect_spans(document)
            body_size = _modal_size(spans)
            blocks: list[TextBlock] = []
            section: list[str] = []
            ocr_pages = 0

            for page_index in range(document.page_count):
                page = document[page_index]
                page_blocks, section = self._parse_page(
                    page, page_index + 1, body_size=body_size, section=section
                )
                text_volume = sum(len(b.text) for b in page_blocks)
                if text_volume < OCR_TRIGGER_CHARS and self._ocr_fallback:
                    ocr_blocks = self._ocr_page(page, page_index + 1)
                    if ocr_blocks:
                        ocr_pages += 1
                        page_blocks = ocr_blocks
                blocks.extend(page_blocks)

            metadata = dict(document.metadata or {})
            return ParsedDocument(
                blocks=tuple(blocks),
                title=(metadata.get("title") or "").strip() or None,
                parser=self.name,
                page_count=document.page_count,
                metadata={
                    "author": metadata.get("author"),
                    "ocr_pages": ocr_pages,
                    "body_font_size": body_size,
                },
            )
        finally:
            document.close()

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _collect_spans(document: Any) -> list[tuple[float, int]]:
        """Gather (font size, character count) for every span in the document."""
        spans: list[tuple[float, int]] = []
        for page in document:
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            spans.append((round(span.get("size", 0.0), 1), len(text)))
        return spans

    def _parse_page(
        self,
        page: Any,
        page_number: int,
        *,
        body_size: float,
        section: list[str],
    ) -> tuple[list[TextBlock], list[str]]:
        """Parse one page into blocks, threading the heading breadcrumb through."""
        width, height = page.rect.width or 1.0, page.rect.height or 1.0
        blocks: list[TextBlock] = []

        table_rects = []
        for table in self._find_tables(page):
            markdown = _table_to_markdown(table.extract())
            if not markdown:
                continue
            table_rects.append(table.bbox)
            blocks.append(
                TextBlock(
                    text=markdown,
                    kind=ChunkKind.TABLE,
                    page_number=page_number,
                    section_path=tuple(section),
                    bbox=_normalise_bbox(table.bbox, width, height),
                )
            )

        for raw in page.get_text("dict").get("blocks", []):
            if raw.get("type") != 0:
                continue
            if _overlaps_any(raw.get("bbox"), table_rects):
                continue  # already captured as a table

            text, size = _block_text_and_size(raw)
            if not text:
                continue

            if size >= body_size * HEADING_SIZE_RATIO and len(text) < 200:
                level = _heading_level(size, body_size)
                section = [*section[: level - 1], text]
                blocks.append(
                    TextBlock(
                        text=text,
                        kind=ChunkKind.HEADING,
                        level=level,
                        page_number=page_number,
                        section_path=tuple(section),
                        bbox=_normalise_bbox(raw.get("bbox"), width, height),
                    )
                )
                continue

            blocks.append(
                TextBlock(
                    text=text,
                    kind=ChunkKind.FOOTNOTE if size < body_size * 0.85 else ChunkKind.PROSE,
                    page_number=page_number,
                    section_path=tuple(section),
                    bbox=_normalise_bbox(raw.get("bbox"), width, height),
                )
            )

        return blocks, section

    @staticmethod
    def _find_tables(page: Any) -> list[Any]:
        """Find tables, tolerating PyMuPDF versions without the finder."""
        try:
            return list(page.find_tables().tables)
        except Exception as exc:  # noqa: BLE001 - table finding is best-effort
            log.debug("table detection unavailable on this page", reason=str(exc))
            return []

    def _ocr_page(self, page: Any, page_number: int) -> list[TextBlock]:
        """OCR a page that yielded no extractable text."""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            log.warning("OCR dependencies unavailable; skipping scanned page", page=page_number)
            return []

        try:
            pixmap = page.get_pixmap(dpi=200)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            text = pytesseract.image_to_string(image).strip()
        except Exception as exc:  # noqa: BLE001 - OCR failure must not fail ingestion
            log.warning("OCR failed for page", page=page_number, reason=str(exc))
            return []

        if not text:
            return []
        return [
            TextBlock(
                text=paragraph.strip(),
                page_number=page_number,
                metadata={"ocr": True},
            )
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]


def _modal_size(spans: Sequence[tuple[float, int]]) -> float:
    """The font size carrying the most characters, i.e. the body text size.

    Weighting by character count rather than span count matters: a document with
    fifty short headings and ten long paragraphs has more heading *spans* than
    body spans, so an unweighted mode would classify the body as headings.

    Example:
        >>> _modal_size([(10.0, 500), (18.0, 20), (18.0, 15)])
        10.0
        >>> _modal_size([])
        11.0
    """
    if not spans:
        return 11.0
    weights: dict[float, int] = {}
    for size, count in spans:
        weights[size] = weights.get(size, 0) + count
    return max(weights.items(), key=lambda item: item[1])[0]


def _heading_level(size: float, body_size: float) -> int:
    """Map a font size onto a heading level 1-4.

    Example:
        >>> _heading_level(22.0, 10.0)
        1
        >>> _heading_level(12.0, 10.0)
        4
    """
    ratio = size / body_size if body_size else 1.0
    if ratio >= 1.8:
        return 1
    if ratio >= 1.5:
        return 2
    if ratio >= 1.3:
        return 3
    return 4


def _block_text_and_size(raw: dict[str, Any]) -> tuple[str, float]:
    """Join a PyMuPDF block's spans and return its dominant font size."""
    parts: list[str] = []
    sizes: list[tuple[float, int]] = []
    for line in raw.get("lines", []):
        line_parts = []
        for span in line.get("spans", []):
            text = span.get("text", "")
            if text.strip():
                line_parts.append(text)
                sizes.append((round(span.get("size", 0.0), 1), len(text.strip())))
        if line_parts:
            parts.append("".join(line_parts).strip())
    return " ".join(parts).strip(), _modal_size(sizes)


def _normalise_bbox(bbox: Any, width: float, height: float) -> BoundingBox | None:
    """Convert a page-space rect into a 0-1 normalised bounding box."""
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = (float(v) for v in bbox)

    def clamp(v: float) -> float:
        """Clamp a normalised coordinate into [0, 1]."""
        return min(max(v, 0.0), 1.0)

    try:
        return BoundingBox(
            x0=clamp(x0 / width),
            y0=clamp(y0 / height),
            x1=clamp(x1 / width),
            y1=clamp(y1 / height),
        )
    except ValueError:
        return None


def _overlaps_any(bbox: Any, rects: Sequence[Any], *, threshold: float = 0.5) -> bool:
    """Whether a block sits mostly inside one of the given rectangles.

    Example:
        >>> _overlaps_any((0, 0, 10, 10), [(0, 0, 10, 10)])
        True
        >>> _overlaps_any((100, 100, 110, 110), [(0, 0, 10, 10)])
        False
    """
    if not bbox or not rects:
        return False
    bx0, by0, bx1, by1 = (float(v) for v in bbox)
    area = max((bx1 - bx0) * (by1 - by0), 1e-6)
    for rect in rects:
        rx0, ry0, rx1, ry1 = (float(v) for v in rect)
        overlap_w = max(0.0, min(bx1, rx1) - max(bx0, rx0))
        overlap_h = max(0.0, min(by1, ry1) - max(by0, ry0))
        if (overlap_w * overlap_h) / area >= threshold:
            return True
    return False


def _table_to_markdown(rows: Sequence[Sequence[Any]]) -> str:
    r"""Render an extracted table as Markdown, skipping empty tables.

    Example:
        >>> _table_to_markdown([["a", "b"], ["1", "2"]])
        '| a | b |\n| --- | --- |\n| 1 | 2 |'
        >>> _table_to_markdown([])
        ''
    """
    cleaned = [[str(cell or "").replace("\n", " ").strip() for cell in row] for row in rows if row]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    header = (cleaned[0] + [""] * width)[:width]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join((row + [""] * width)[:width]) + " |" for row in cleaned[1:])
    return "\n".join(lines)


pdf_parser = register_parser(PdfParser())
