"""One facade over memory content and its derived indexes."""

from __future__ import annotations

import builtins
from collections.abc import Iterable
from dataclasses import dataclass, field

from memory.evolution.semantic_abstraction_engine import factual_evidence_eligible
from memory.graph.graph_builder import GraphBuilder
from memory.graph.graph_repository import ChromaGraphRepository, GraphRepository
from memory.memory_note import MemoryNote
from memory.memory_repository import ChromaMemoryRepository, MemoryRepository
from memory.memory_types import MemoryType


@dataclass(slots=True)
class MemoryIndex(MemoryRepository):
    """Coordinate repository, graph, temporal, and metadata access.

    Memory content remains solely in ``repository``. The graph stores only
    derived node/edge metadata keyed by memory IDs.
    """

    repository: MemoryRepository
    graph_repository: GraphRepository | None = field(default_factory=GraphRepository)
    _graph_initialized: bool = field(default=False, init=False, repr=False)
    factual_only: bool = False
    source_conversation_id: str | None = None

    def _visible(self, note: MemoryNote) -> bool:
        return ((not self.factual_only or factual_evidence_eligible(note)) and (not self.source_conversation_id
            or note.context.get("source_conversation_id", self.source_conversation_id) == self.source_conversation_id))

    def __post_init__(self) -> None:
        if self.graph_repository is None:
            self._graph_initialized = True

    @classmethod
    def for_repository(cls, repository: MemoryRepository) -> MemoryIndex:
        """Use a persistent graph mirror when the content repository is Chroma."""
        graph = (
            ChromaGraphRepository(str(repository.path))
            if isinstance(repository, ChromaMemoryRepository)
            else GraphRepository()
        )
        return cls(repository, graph)

    @property
    def capabilities(self) -> dict[str, bool]:
        """Return available index capabilities for route diagnostics."""
        return {
            "vector": True,
            "graph": self.graph_repository is not None,
            "temporal": True,
            "metadata": True,
        }

    def add(self, note: MemoryNote) -> MemoryNote:
        self._validate_procedure(note)
        saved = self.repository.add(note)
        self._index_note(saved)
        return saved

    def update(self, note: MemoryNote) -> MemoryNote:
        self._validate_procedure(note)
        saved = self.repository.update(note)
        self._index_note(saved)
        return saved

    def get(self, note_id: str, memory_type: MemoryType | None = None) -> MemoryNote | None:
        note = self.repository.get(note_id, memory_type)
        return note if note is None or self._visible(note) else None

    def list(self, memory_type: MemoryType | None = None) -> builtins.list[MemoryNote]:
        return [note for note in self.repository.list(memory_type) if self._visible(note)]

    def query(
        self,
        embedding: Iterable[float],
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> builtins.list[tuple[MemoryNote, float]]:
        excluded = sum(not self._visible(note) for note in self.repository.list(memory_type)) if self.factual_only or self.source_conversation_id else 0
        rows = self.repository.query(embedding, memory_type, limit + excluded)
        return [(note, score) for note, score in rows if self._visible(note)][:limit]

    def _sync_graph(self) -> None:
        notes = self.repository.list()
        for note in notes:
            self._validate_procedure(note)
        if self.graph_repository is None:
            return
        known = {node.memory_id for node in self.graph_repository.nodes.values()}
        for note in notes:
            if note.id not in known:
                self._index_note(note)
                known.add(note.id)
        self._graph_initialized = True

    def ensure_graph_index(self) -> None:
        """Lazily synchronize pre-existing repository rows once before graph retrieval."""

        if not self._graph_initialized:
            self._sync_graph()

    def _index_note(self, note: MemoryNote) -> None:
        if self.graph_repository is None:
            return
        GraphBuilder(self.graph_repository).index_note(self.repository, note)

    @staticmethod
    def _validate_procedure(note: MemoryNote) -> None:
        if note.memory_type is not MemoryType.PROCEDURAL:
            return
        if not (note.context.get("verified_success") is True or note.context.get("explicit_import") is True):
            raise ValueError("Procedural memory requires verified successful execution or explicit import provenance.")
