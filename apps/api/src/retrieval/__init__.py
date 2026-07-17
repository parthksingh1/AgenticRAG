"""Retrieval strategies and the hybrid orchestrator."""

from src.retrieval.base import InMemoryRetriever, Retriever, matches_filters
from src.retrieval.corrective import (
    CragAssessment,
    CragThresholds,
    NoWebSearch,
    RetrievalEvaluator,
    TavilyWebSearch,
)
from src.retrieval.fusion import (
    deduplicate_by_document,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from src.retrieval.graph import GraphEntity, GraphRelation, GraphRetriever
from src.retrieval.hybrid import HybridConfig, HybridRetriever
from src.retrieval.rerank import (
    CohereReranker,
    CrossEncoderReranker,
    IdentityReranker,
    Reranker,
    ScriptedReranker,
)
from src.retrieval.rewrite import (
    QueryRewriter,
    RewriteResult,
    looks_multi_hop,
    looks_time_sensitive,
)
from src.retrieval.types import (
    CragVerdict,
    FusionConfig,
    RetrievalRequest,
    RetrievalResult,
    RetrievalSource,
    RetrievedChunk,
)

__all__ = [
    "CohereReranker",
    "CragAssessment",
    "CragThresholds",
    "CragVerdict",
    "CrossEncoderReranker",
    "FusionConfig",
    "GraphEntity",
    "GraphRelation",
    "GraphRetriever",
    "HybridConfig",
    "HybridRetriever",
    "IdentityReranker",
    "InMemoryRetriever",
    "NoWebSearch",
    "QueryRewriter",
    "Reranker",
    "RetrievalEvaluator",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievalSource",
    "RetrievedChunk",
    "Retriever",
    "RewriteResult",
    "ScriptedReranker",
    "TavilyWebSearch",
    "deduplicate_by_document",
    "looks_multi_hop",
    "looks_time_sensitive",
    "matches_filters",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
]
