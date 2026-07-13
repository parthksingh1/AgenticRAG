"""Parser protocol, registry and format detection.

A parser's job is narrow: turn bytes into an ordered list of
:class:`~src.ingestion.types.TextBlock` with structure preserved. It does not
chunk, embed or store. That separation is what lets the chunker promise a table
is never split — the parser has already told it where the table is.

Format detection prefers content sniffing over the filename, because uploads
routinely arrive with a wrong or missing extension and a PDF parsed as text
produces silent garbage rather than an error.

Example:
    >>> detect_format(b"%PDF-1.7 ...", filename="mystery.txt")
    'pdf'
    >>> detect_format(b"plain words", filename="notes.md")
    'markdown'
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.core.errors import UnsupportedMediaTypeError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.ingestion.types import ParsedDocument

log = get_logger(__name__)

#: Leading bytes that identify a format regardless of what the filename claims.
_MAGIC_BYTES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF", "pdf"),
    (b"\x89PNG\r\n\x1a\n", "image"),
    (b"\xff\xd8\xff", "image"),
    (b"GIF87a", "image"),
    (b"GIF89a", "image"),
    (b"BM", "image"),
    (b"II*\x00", "image"),
    (b"MM\x00*", "image"),
)

#: Extension -> format, used when the bytes are not self-identifying.
_EXTENSION_FORMATS: dict[str, str] = {
    "pdf": "pdf",
    "docx": "docx",
    "doc": "docx",
    "pptx": "pptx",
    "ppt": "pptx",
    "xlsx": "xlsx",
    "xls": "xlsx",
    "csv": "csv",
    "tsv": "csv",
    "html": "html",
    "htm": "html",
    "md": "markdown",
    "markdown": "markdown",
    "txt": "text",
    "text": "text",
    "log": "text",
    "json": "text",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "gif": "image",
    "bmp": "image",
    "tif": "image",
    "tiff": "image",
    "webp": "image",
}

#: ZIP-based Office formats share a magic number, so they are disambiguated by
#: the entry names inside the archive.
_OOXML_MARKERS: tuple[tuple[bytes, str], ...] = (
    (b"word/", "docx"),
    (b"ppt/", "pptx"),
    (b"xl/", "xlsx"),
)


class Parser(ABC):
    """Turns raw bytes of one format into structured text blocks."""

    #: Stable parser name, recorded on the document version for reproducibility.
    name: str
    #: Formats this parser claims, as returned by :func:`detect_format`.
    formats: frozenset[str]

    @abstractmethod
    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        """Parse ``data`` into a structured document.

        Raises:
            IngestionError: when the bytes cannot be read as this format.
        """


_REGISTRY: dict[str, Parser] = {}


def register_parser(parser: Parser) -> Parser:
    """Register a parser for each format it claims.

    Raises:
        ValueError: on a duplicate registration, which would otherwise make the
            active parser depend on import order.
    """
    for fmt in parser.formats:
        if fmt in _REGISTRY:
            msg = f"format {fmt!r} already handled by {_REGISTRY[fmt].name!r}"
            raise ValueError(msg)
        _REGISTRY[fmt] = parser
    return parser


def get_parser(fmt: str) -> Parser:
    """Return the parser for a format.

    Raises:
        UnsupportedMediaTypeError: when nothing handles the format, so the API
            answers 415 rather than failing deep inside a Celery task.
    """
    parser = _REGISTRY.get(fmt)
    if parser is None:
        supported = ", ".join(sorted(_REGISTRY)) or "none"
        raise UnsupportedMediaTypeError(
            f"No parser for format {fmt!r}. Supported: {supported}.",
            details={"format": fmt, "supported": sorted(_REGISTRY)},
        )
    return parser


def supported_formats() -> tuple[str, ...]:
    """Every format with a registered parser."""
    return tuple(sorted(_REGISTRY))


def detect_format(data: bytes, *, filename: str | None = None) -> str:
    r"""Identify a document's format from its bytes, then its filename.

    Content wins over filename: an upload named ``.txt`` that is really a PDF
    must be parsed as a PDF, because parsing it as text yields plausible-looking
    binary noise that would be indexed and cited as if it were content.

    Example:
        >>> detect_format(b"%PDF-1.4")
        'pdf'
        >>> detect_format(b"col_a,col_b\n1,2\n", filename="data.csv")
        'csv'
        >>> detect_format(b"<html><body>hi</body></html>")
        'html'
        >>> detect_format(b"just words")
        'text'
    """
    head = data[:2048]

    for magic, fmt in _MAGIC_BYTES:
        if head.startswith(magic):
            return fmt

    if head.startswith(b"PK\x03\x04"):
        # OOXML containers all look like ZIPs; look inside for the giveaway.
        window = data[:8192]
        for marker, fmt in _OOXML_MARKERS:
            if marker in window:
                return fmt
        return "docx"

    lowered = head.lstrip().lower()
    if lowered.startswith((b"<!doctype html", b"<html", b"<?xml")) or b"<body" in lowered:
        return "html"

    extension = (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""
    if extension in _EXTENSION_FORMATS:
        return _EXTENSION_FORMATS[extension]

    return "text"


def parse_bytes(data: bytes, *, filename: str | None = None) -> ParsedDocument:
    """Detect the format and parse, the single entry point ingestion uses.

    Raises:
        UnsupportedMediaTypeError: when no parser handles the detected format.
    """
    fmt = detect_format(data, filename=filename)
    parser = get_parser(fmt)
    log.debug("parsing document", format=fmt, parser=parser.name, bytes=len(data))
    return parser.parse(data, filename=filename)
