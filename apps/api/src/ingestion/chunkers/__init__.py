"""Chunking strategies.

Importing this package registers the strategies that need no runtime
dependencies (``layout_aware`` and ``fixed``). The two model-backed strategies —
``semantic`` and ``late`` — are registered by their ``build_*`` factories once a
concrete embedder or encoder is available, because a strategy that silently
falls back to a stub would corrupt an eval run without failing it.
"""

from src.ingestion.chunkers.base import (
    Chunker,
    available_chunkers,
    get_chunker,
    register_chunker,
    split_oversized_text,
    split_sentences,
)
from src.ingestion.chunkers.fixed import FixedSizeChunker, fixed_size_chunker
from src.ingestion.chunkers.late import LateChunker, build_late_chunker
from src.ingestion.chunkers.layout import LayoutAwareChunker, layout_aware_chunker
from src.ingestion.chunkers.semantic import SemanticChunker, build_semantic_chunker

__all__ = [
    "Chunker",
    "FixedSizeChunker",
    "LateChunker",
    "LayoutAwareChunker",
    "SemanticChunker",
    "available_chunkers",
    "build_late_chunker",
    "build_semantic_chunker",
    "fixed_size_chunker",
    "get_chunker",
    "layout_aware_chunker",
    "register_chunker",
    "split_oversized_text",
    "split_sentences",
]
