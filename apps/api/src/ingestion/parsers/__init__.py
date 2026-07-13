"""Document parsers.

Importing this package registers every parser, which is what makes
:func:`~src.ingestion.parsers.base.parse_bytes` able to handle any supported
format without the caller naming one.
"""

from src.ingestion.parsers.base import (
    Parser,
    detect_format,
    get_parser,
    parse_bytes,
    register_parser,
    supported_formats,
)
from src.ingestion.parsers.image import ImageParser
from src.ingestion.parsers.office import DocxParser, PptxParser, XlsxParser
from src.ingestion.parsers.pdf import PdfParser
from src.ingestion.parsers.text import CsvParser, MarkdownParser, TextParser
from src.ingestion.parsers.web import HtmlParser

__all__ = [
    "CsvParser",
    "DocxParser",
    "HtmlParser",
    "ImageParser",
    "MarkdownParser",
    "Parser",
    "PdfParser",
    "PptxParser",
    "TextParser",
    "XlsxParser",
    "detect_format",
    "get_parser",
    "parse_bytes",
    "register_parser",
    "supported_formats",
]
