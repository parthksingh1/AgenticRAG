r"""Plain text, Markdown and delimited-data parsers.

These carry no external dependencies, which makes them the parsers the test
suite and the offline demo lean on. The Markdown parser is more than a
passthrough: it recovers heading levels, fenced code blocks and pipe tables as
distinct blocks, so layout-aware chunking works on Markdown exactly as it does
on a PDF.

Example:
    >>> doc = MarkdownParser().parse(b"# Title\n\nSome prose.")
    >>> [(b.kind.value, b.text) for b in doc.blocks]
    [('heading', 'Title'), ('prose', 'Some prose.')]
"""

from __future__ import annotations

import csv
import io
import re

from src.core.errors import IngestionError
from src.ingestion.parsers.base import Parser, register_parser
from src.ingestion.types import ParsedDocument, TextBlock
from src.models.document import ChunkKind

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^\s*```")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+\.)\s+")

#: Rows beyond this are summarised rather than emitted, so a million-row CSV
#: does not become a million chunks.
MAX_CSV_ROWS = 5_000


def decode(data: bytes) -> str:
    """Decode bytes to text, trying the encodings uploads actually use.

    Falls back to lossy UTF-8 rather than raising: a document with a handful of
    undecodable bytes is still worth indexing, and refusing it would be a
    frustrating failure for a user who cannot control the file's encoding.

    Example:
        >>> decode(b"hello")
        'hello'
        >>> decode("café".encode("latin-1"))
        'café'
    """
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class TextParser(Parser):
    """Splits plain text into paragraph blocks."""

    name = "text"
    formats = frozenset({"text"})

    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        """Split on blank lines into paragraphs."""
        text = decode(data)
        blocks = tuple(
            TextBlock(text=paragraph.strip())
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip()
        )
        return ParsedDocument(
            blocks=blocks,
            title=_title_from_filename(filename),
            parser=self.name,
        )


class MarkdownParser(Parser):
    """Recovers headings, code fences, tables and lists from Markdown."""

    name = "markdown"
    formats = frozenset({"markdown"})

    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        r"""Parse Markdown into typed blocks.

        Example:
            >>> md = b"## Setup\n\n```py\nx = 1\n```\n\n| a | b |\n| - | - |\n"
            >>> [b.kind.value for b in MarkdownParser().parse(md).blocks]
            ['heading', 'code', 'table']
        """
        lines = decode(data).splitlines()
        blocks: list[TextBlock] = []
        buffer: list[str] = []
        section: list[str] = []
        mode = "prose"

        def flush(kind: ChunkKind = ChunkKind.PROSE) -> None:
            """Emit the buffer as one block."""
            body = "\n".join(buffer).strip()
            buffer.clear()
            if body:
                blocks.append(TextBlock(text=body, kind=kind, section_path=tuple(section)))

        for line in lines:
            if _FENCE.match(line):
                if mode == "code":
                    flush(ChunkKind.CODE)
                    mode = "prose"
                else:
                    flush()
                    mode = "code"
                continue

            if mode == "code":
                buffer.append(line)
                continue

            heading = _HEADING.match(line)
            if heading:
                flush(_kind_for(mode))
                mode = "prose"
                level = len(heading.group(1))
                title = heading.group(2).strip()
                section = [*section[: level - 1], title]
                blocks.append(
                    TextBlock(
                        text=title,
                        kind=ChunkKind.HEADING,
                        level=level,
                        section_path=tuple(section),
                    )
                )
                continue

            if _TABLE_ROW.match(line):
                if mode != "table":
                    flush(_kind_for(mode))
                    mode = "table"
                buffer.append(line)
                continue

            if _LIST_ITEM.match(line):
                if mode != "list":
                    flush(_kind_for(mode))
                    mode = "list"
                buffer.append(line)
                continue

            if not line.strip():
                flush(_kind_for(mode))
                mode = "prose"
                continue

            if mode in ("table", "list"):
                flush(_kind_for(mode))
                mode = "prose"
            buffer.append(line)

        flush(_kind_for(mode))
        return ParsedDocument(
            blocks=tuple(blocks),
            title=_markdown_title(blocks) or _title_from_filename(filename),
            parser=self.name,
        )


class CsvParser(Parser):
    """Renders delimited data as Markdown tables.

    Tabular data is emitted as a table block rather than as prose so the chunker
    keeps it intact and the SQL-analytics MCP server can recognise it. Very large
    files are truncated with an explicit note, because a hundred thousand rows of
    embedded numbers is expensive to index and near-useless to retrieve.
    """

    name = "csv"
    formats = frozenset({"csv"})

    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        """Parse CSV or TSV into a header block plus table blocks."""
        text = decode(data)
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(io.StringIO(text), dialect)
        try:
            rows = list(reader)
        except csv.Error as exc:
            raise IngestionError(f"malformed delimited data: {exc}") from exc

        if not rows:
            return ParsedDocument(blocks=(), parser=self.name)

        header, body = rows[0], rows[1:]
        truncated = len(body) > MAX_CSV_ROWS
        shown = body[:MAX_CSV_ROWS]

        blocks: list[TextBlock] = [
            TextBlock(
                text=f"Columns: {', '.join(header)}. Rows: {len(body)}.",
                kind=ChunkKind.CAPTION,
                metadata={"columns": header, "row_count": len(body)},
            )
        ]
        # Emit in windows so one table block stays a sensible retrieval unit.
        window = 50
        for start in range(0, len(shown), window):
            blocks.append(
                TextBlock(
                    text=_markdown_table(header, shown[start : start + window]),
                    kind=ChunkKind.TABLE,
                    metadata={"row_offset": start},
                )
            )
        if truncated:
            blocks.append(
                TextBlock(
                    text=f"[{len(body) - MAX_CSV_ROWS} further rows omitted from the index.]",
                    kind=ChunkKind.CAPTION,
                )
            )

        return ParsedDocument(
            blocks=tuple(blocks),
            title=_title_from_filename(filename),
            parser=self.name,
            metadata={"columns": header, "row_count": len(body), "truncated": truncated},
        )


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Render rows as a Markdown table.

    Example:
        >>> print(_markdown_table(["a", "b"], [["1", "2"]]))
        | a | b |
        | --- | --- |
        | 1 | 2 |
    """
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend(
        "| " + " | ".join(_clean_cell(cell) for cell in _pad(row, len(header))) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _pad(row: list[str], width: int) -> list[str]:
    """Pad or trim a row to the header width, so ragged CSVs still render."""
    return (row + [""] * width)[:width]


def _clean_cell(cell: str) -> str:
    """Escape pipes and collapse newlines so one cell cannot break the table."""
    return cell.replace("|", "\\|").replace("\n", " ").strip()


def _kind_for(mode: str) -> ChunkKind:
    """Map the parser's line mode onto a chunk kind."""
    return {
        "code": ChunkKind.CODE,
        "table": ChunkKind.TABLE,
        "list": ChunkKind.LIST,
    }.get(mode, ChunkKind.PROSE)


def _markdown_title(blocks: list[TextBlock]) -> str | None:
    """The first level-1 heading, if the document has one."""
    return next(
        (b.text for b in blocks if b.kind is ChunkKind.HEADING and b.level == 1),
        None,
    )


def _title_from_filename(filename: str | None) -> str | None:
    """Derive a readable title from a filename.

    Example:
        >>> _title_from_filename("q3_sales_report.pdf")
        'Q3 Sales Report'
        >>> _title_from_filename(None) is None
        True
    """
    if not filename:
        return None
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    words = re.split(r"[_\-\s]+", stem)
    return " ".join(word.capitalize() for word in words if word) or None


text_parser = register_parser(TextParser())
markdown_parser = register_parser(MarkdownParser())
csv_parser = register_parser(CsvParser())
