"""Chunker protocol, registry and shared text-splitting helpers.

Every strategy implements the same tiny interface, so swapping one per tenant is
a config change rather than a code change. The helpers below are shared because
the invariants the property tests assert (no text loss, contiguous ordinals,
size ceilings respected) are only trustworthy if every strategy splits text the
same way.

Example:
    >>> from src.ingestion.chunkers.base import split_sentences
    >>> split_sentences("One. Two! Three?")
    ['One.', 'Two!', 'Three?']
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.ingestion.types import ChunkDraft, ChunkingConfig, ParsedDocument

#: Sentence boundary: terminator, optional closing quote/bracket, then whitespace
#: followed by something that can start a sentence. Deliberately conservative —
#: over-splitting costs a little recall, under-splitting breaks the size ceiling.
_SENTENCE_END = re.compile(r'(?<=[.!?])["\')\]]*\s+(?=["\'(\[]*[A-Z0-9])')

#: Abbreviations that end in a period but do not end a sentence.
_ABBREVIATIONS = frozenset(
    {
        "e.g.", "i.e.", "etc.", "cf.", "vs.", "fig.", "eq.", "al.", "no.",
        "dr.", "mr.", "mrs.", "ms.", "prof.", "st.", "approx.", "ref.", "sec.",
    }
)  # fmt: skip


@runtime_checkable
class Chunker(Protocol):
    """Splits a parsed document into ordered, retrievable chunks.

    Implementations must be pure: same document and config in, same chunks out,
    with no I/O and no hidden state. That is what makes the property tests
    meaningful and lets ingestion be retried safely.
    """

    name: str

    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> list[ChunkDraft]:
        """Return chunks in reading order with contiguous ordinals from zero."""
        ...


_REGISTRY: dict[str, Chunker] = {}


def register_chunker(chunker: Chunker) -> Chunker:
    """Register a chunker under its ``name`` and return it unchanged.

    Raises:
        ValueError: if the name is already taken, which would otherwise make the
            active strategy depend on module import order.
    """
    if chunker.name in _REGISTRY:
        msg = f"chunker {chunker.name!r} is already registered"
        raise ValueError(msg)
    _REGISTRY[chunker.name] = chunker
    return chunker


def get_chunker(name: str) -> Chunker:
    """Look up a registered chunker by strategy name.

    Raises:
        KeyError: if no chunker is registered under that name.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        msg = f"unknown chunking strategy {name!r}; registered: {available}"
        raise KeyError(msg) from None


def available_chunkers() -> tuple[str, ...]:
    """Names of every registered chunking strategy."""
    return tuple(sorted(_REGISTRY))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping terminators attached.

    Not a full NLP sentence splitter — it is a deterministic, dependency-free
    approximation that avoids the common abbreviation traps. The semantic
    chunker refines these boundaries with embeddings; everything else just needs
    a reasonable place to cut.

    Example:
        >>> split_sentences("See Fig. 2 for detail. It shows growth.")
        ['See Fig. 2 for detail.', 'It shows growth.']
        >>> split_sentences("")
        []
    """
    stripped = text.strip()
    if not stripped:
        return []

    pieces = _SENTENCE_END.split(stripped)
    sentences: list[str] = []
    for piece in pieces:
        candidate = piece.strip()
        if not candidate:
            continue
        # Re-join when the previous fragment ended on a known abbreviation.
        if sentences and sentences[-1].split()[-1].lower() in _ABBREVIATIONS:
            sentences[-1] = f"{sentences[-1]} {candidate}"
        else:
            sentences.append(candidate)
    return sentences


def split_oversized_text(text: str, *, max_chars: int, overlap_chars: int = 0) -> list[str]:
    """Hard-split text that no softer boundary could bring under ``max_chars``.

    Prefers sentence boundaries, falls back to whitespace, and only cuts
    mid-word when a single token genuinely exceeds the limit. Guaranteed to
    terminate and to preserve every character of the input across the returned
    pieces (ignoring the deliberate overlap).

    Args:
        text: The text to split.
        max_chars: Hard ceiling for each returned piece.
        overlap_chars: Characters of trailing context repeated at the start of
            the next piece. Must be smaller than ``max_chars``.

    Example:
        >>> split_oversized_text("abcdefghij", max_chars=4)
        ['abcd', 'efgh', 'ij']
    """
    if max_chars <= 0:
        msg = "max_chars must be positive"
        raise ValueError(msg)
    if overlap_chars >= max_chars:
        msg = "overlap_chars must be smaller than max_chars"
        raise ValueError(msg)
    if len(text) <= max_chars:
        return [text] if text else []

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            end = _best_cut(text, start=start, end=end)
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return pieces


def _best_cut(text: str, *, start: int, end: int) -> int:
    """Find the latest natural boundary at or before ``end``.

    Looks back at most a quarter of the window so a document with no whitespace
    degrades to a hard cut rather than to tiny chunks.
    """
    window_floor = start + (end - start) * 3 // 4
    for boundary in (". ", "! ", "? ", "\n", " "):
        cut = text.rfind(boundary, window_floor, end)
        if cut != -1:
            return cut + len(boundary)
    return end


def apply_overlap(previous: str, current: str, *, overlap_chars: int) -> str:
    """Prefix ``current`` with the tail of ``previous`` for retrieval continuity.

    The overlap is cut back to a whitespace boundary so a chunk never begins
    mid-word, which reads badly in the citation UI.

    Example:
        >>> apply_overlap("the quick brown fox", "jumps over", overlap_chars=9)
        'brown fox jumps over'
        >>> apply_overlap("anything", "current", overlap_chars=0)
        'current'
    """
    if overlap_chars <= 0 or not previous:
        return current
    cut = max(len(previous) - overlap_chars, 0)
    tail = previous[cut:]
    # Only trim when the cut landed inside a word; a cut on a word boundary is
    # already clean and trimming it would silently shorten the overlap.
    if cut > 0 and not previous[cut - 1].isspace() and not tail[:1].isspace():
        space = tail.find(" ")
        tail = tail[space + 1 :] if space != -1 else ""
    tail = tail.strip()
    return f"{tail} {current}" if tail else current


def normalise_whitespace(text: str) -> str:
    r"""Collapse runs of blank lines and trailing spaces without losing paragraphs.

    Example:
        >>> normalise_whitespace("a  \n\n\n\nb   ")
        'a\n\nb'
    """
    text = re.sub(r"[ \t]+(\n)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
