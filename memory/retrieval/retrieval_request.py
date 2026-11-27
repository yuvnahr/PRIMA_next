"""Retrieval request model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from typing import Any

from memory.embedding_pipeline import get_embedding_pipeline
from memory.memory_index import MemoryIndex
from memory.memory_note import MemoryNote
from memory.memory_repository import MemoryRepository
from memory.memory_types import MemoryType, RetrievalWindow
from memory.retrieval.query_analysis import ExpandedQuery, QueryAnalysis, QueryRewrite


def observation_time(value: datetime | str, *, end_of_day: bool = False) -> datetime:
    if isinstance(value, str):
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if end_of_day and len(value) == 10:
            instant = datetime.combine(instant.date(), time.max)
    else:
        instant = value
    return instant.replace(tzinfo=instant.tzinfo or UTC).astimezone(UTC)


def effective_time(note: MemoryNote) -> datetime:
    return observation_time(note.context.get("effective_from", note.timestamp))


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
    session_id: str = ""
    reference_time: datetime | None = None
    historical: bool = False

    def __post_init__(self) -> None:
        if self.reference_time is not None:
            object.__setattr__(self, "reference_time", observation_time(self.reference_time, end_of_day=True))

    def as_of_view(self, repository: MemoryRepository) -> MemoryRepository:
        cutoff = self.reference_time
        if cutoff is None:
            return repository
        if (
            isinstance(repository, MemoryIndex)
            and repository.observed_through is not None
            and repository.observed_through <= cutoff
        ):
            return repository
        notes = {row.id: row for row in repository.list()}

        def available(note: MemoryNote, visiting: set[str]) -> bool:
            if (
                note.id in visiting
                or note.retrieval_metadata.get("observation_time_unknown")
                and "source_timestamp" not in note.context
            ):
                return False
            observed = note.context.get("source_timestamp", note.timestamp)
            if not isinstance(observed, (str, datetime)):
                return False
            if note.context.get("timestamp_precision") == "date":
                observed = str(observed)[:10]
            if observation_time(observed, end_of_day=True) > cutoff:
                return False
            parents = note.lineage.parent_ids
            if note.context.get("source") == "memory_evolution" and not parents:
                return False
            return all(key in notes and available(notes[key], visiting | {note.id}) for key in parents)

        return MemoryIndex(
            repository,
            graph_repository=None,
            observed_through=cutoff,
            visible_ids=frozenset(key for key, note in notes.items() if available(note, set())),
        )

    def embedding(self) -> tuple[float, ...]:
        return self.query_embedding or get_embedding_pipeline().embed_query(self.query).vector

    def lexical_query(self) -> str:
        return self.expanded_query.text if self.expanded_query is not None else self.query
