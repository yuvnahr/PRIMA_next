"""Retrieval request model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory.memory_note import stable_embedding
from memory.memory_types import MemoryType, RetrievalWindow


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    query_embedding: tuple[float, ...] | None = None
    memory_types: tuple[MemoryType, ...] = tuple(MemoryType)
    top_k: int = 5
    state_filter: dict[str, Any] = field(default_factory=dict)
    affective_context: dict[str, Any] = field(default_factory=dict)
    temporal_window: RetrievalWindow = RetrievalWindow.LONG_TERM

    def embedding(self) -> tuple[float, ...]:
        return self.query_embedding or tuple(stable_embedding(self.query))
