"""Retrieval request model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory.embedding_pipeline import get_embedding_pipeline
from memory.memory_types import MemoryType, RetrievalWindow
from memory.retrieval.query_analysis import ExpandedQuery, QueryAnalysis, QueryRewrite


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    query_embedding: tuple[float, ...] | None = None
    analyzed_query: QueryAnalysis | None = None
    expanded_query: ExpandedQuery | None = None
    query_rewrite: QueryRewrite | None = None
    memory_types: tuple[MemoryType, ...] = tuple(MemoryType)
    top_k: int = 5
    state_filter: dict[str, Any] = field(default_factory=dict)
    affective_context: dict[str, Any] = field(default_factory=dict)
    temporal_window: RetrievalWindow = RetrievalWindow.LONG_TERM
    profile: str = "prima_full"
    task_kind: str = "factual_qa"
    required_reranker_backend: str | None = None

    def embedding(self) -> tuple[float, ...]:
        return self.query_embedding or get_embedding_pipeline().embed_query(self.query).vector

    def lexical_query(self) -> str:
        return self.expanded_query.text if self.expanded_query is not None else self.query



