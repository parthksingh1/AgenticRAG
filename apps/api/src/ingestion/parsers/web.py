"""HTML parsing.

Web pages are mostly not content. Navigation, cookie banners, related-article
rails and footers make up the bulk of the markup, and indexing them poisons
retrieval with text that matches everything and answers nothing.

trafilatura does the extraction because it is specifically trained on that
boilerplate-removal problem. Its Markdown output is then re-parsed by the
Markdown parser, which recovers headings and tables — so an HTML page ends up
with the same block structure as a PDF, and downstream code needs no special
case. When trafilatura is unavailable or finds nothing, a conservative built-in
stripper takes over rather than failing the ingestion.

Example:
    >>> HtmlParser().formats == frozenset({"html"})
    True
"""

from __future__ import annotations

import html as html_module
import re

from src.core.logging import get_logger
from src.ingestion.parsers.base import Parser, register_parser
from src.ingestion.parsers.text import MarkdownParser, decode
from src.ingestion.types import ParsedDocument, TextBlock

log = get_logger(__name__)

#: Elements whose contents are never content.
_STRIP_ELEMENTS = ("script", "style", "noscript", "template", "svg", "iframe")

_TAG = re.compile(r"<[^>]+>")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_BLOCK_BOUNDARY = re.compile(
    r"</(?:p|div|section|article|h[1-6]|li|tr|blockquote|pre)\s*>", re.IGNORECASE
)


class HtmlParser(Parser):
    """Extracts the main content of an HTML page."""

    name = "trafilatura"
    formats = frozenset({"html"})

    def __init__(self, *, markdown_parser: MarkdownParser | None = None) -> None:
        """Create the parser, optionally injecting the Markdown re-parser."""
        self._markdown = markdown_parser or MarkdownParser()

    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        """Extract content, preferring trafilatura and falling back to stripping."""
        raw = decode(data)
        title = _extract_title(raw)

        markdown = self._extract_with_trafilatura(raw)
        if markdown:
            parsed = self._markdown.parse(markdown.encode("utf-8"), filename=filename)
            return ParsedDocument(
                blocks=parsed.blocks,
                title=title or parsed.title,
                parser=self.name,
                metadata={"extractor": "trafilatura", "source": filename},
            )

        log.info("trafilatura extracted nothing; falling back to tag stripping")
        blocks = tuple(
            TextBlock(text=paragraph) for paragraph in _strip_tags(raw) if paragraph.strip()
        )
        return ParsedDocument(
            blocks=blocks,
            title=title,
            parser=f"{self.name}-fallback",
            metadata={"extractor": "strip_tags", "source": filename},
        )

    @staticmethod
    def _extract_with_trafilatura(raw: str) -> str | None:
        """Run trafilatura, returning Markdown or None when it is unavailable."""
        try:
            import trafilatura
        except ImportError:
            log.warning("trafilatura not installed; using the fallback extractor")
            return None

        try:
            return trafilatura.extract(
                raw,
                output_format="markdown",
                include_tables=True,
                include_links=False,
                include_comments=False,
                favor_precision=True,
            )
        except Exception as exc:  # noqa: BLE001 - extraction is best-effort
            log.warning("trafilatura failed; using the fallback extractor", reason=str(exc))
            return None


def _extract_title(raw: str) -> str | None:
    """Read the document title from the ``<title>`` element.

    Example:
        >>> _extract_title("<html><head><title>Hello &amp; welcome</title></head></html>")
        'Hello & welcome'
        >>> _extract_title("<html></html>") is None
        True
    """
    match = _TITLE.search(raw)
    if not match:
        return None
    return html_module.unescape(_TAG.sub("", match.group(1))).strip() or None


def _strip_tags(raw: str) -> list[str]:
    """Strip markup into paragraphs, dropping scripts and styles entirely.

    A deliberately conservative fallback: it keeps too much rather than too
    little, because losing the page's actual content is worse than indexing some
    navigation text alongside it.

    Example:
        >>> _strip_tags("<p>One</p><script>evil()</script><p>Two</p>")
        ['One', 'Two']
    """
    cleaned = raw
    for element in _STRIP_ELEMENTS:
        cleaned = re.sub(
            rf"<{element}\b.*?</{element}\s*>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL
        )

    cleaned = _BLOCK_BOUNDARY.sub("\n\n", cleaned)
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = _TAG.sub(" ", cleaned)
    cleaned = html_module.unescape(cleaned)

    paragraphs = []
    for chunk in cleaned.split("\n\n"):
        collapsed = re.sub(r"[ \t\r\f\v]+", " ", chunk).strip()
        if collapsed:
            paragraphs.append(collapsed)
    return paragraphs


html_parser = register_parser(HtmlParser())
