"""Build graph links from memory context chains."""

from __future__ import annotations

from memory.graph.graph_edge import GraphEdge
from memory.graph.graph_node import GraphNode
from memory.graph.graph_repository import GraphRepository
from memory.memory_note import MemoryNote, calculate_smart_overlap
from memory.memory_repository import MemoryRepository
from memory.memory_types import MemoryType


class GraphBuilder:
    def __init__(self, graph_repository: GraphRepository) -> None:
        self.graph_repository = graph_repository

    def rebuild_for_type(self, repository: MemoryRepository, memory_type: MemoryType) -> GraphRepository:
        notes = [note for note in repository.list(memory_type) if not note.context.get("superseded")]
        for note in notes:
            self.graph_repository.add_node(self._node(note))
        for index, note in enumerate(notes):
            self._pairs(note, notes[index + 1 :])
        for node in tuple(self.graph_repository.nodes.values()):
            if node.labels[:1] == (memory_type.value,) and repository.get(node.memory_id) is None:
                self.graph_repository.remove_node(node.id)
        return self.graph_repository

    def _node(self, note: MemoryNote) -> GraphNode:
        metadata: dict[str, object] = {"keywords": list(note.keywords)}
        if note.context.get("superseded"):
            metadata["superseded"] = True
        return GraphNode(
            id=f"node_{note.id}",
            memory_id=note.id,
            labels=(note.memory_type.value, note.memory_level.value),
            metadata=metadata,
        )

    def _pairs(self, note: MemoryNote, others: list[MemoryNote]) -> None:
        for other in others:
            if other.context.get("superseded") or other.id == note.id:
                continue
            left, right = sorted((note, other), key=lambda item: item.id)
            overlap = calculate_smart_overlap(left.keywords, right.keywords)
            edge = GraphEdge(
                f"node_{left.id}",
                f"node_{right.id}",
                "shared_context",
                min(1.0, 0.35 + overlap * 0.2),
                {"producer": "graph_builder", "overlap": overlap},
            )
            if overlap:
                self.graph_repository.add_edge(edge)
            elif edge.key in self.graph_repository.edges:
                self.graph_repository.remove_edge(edge)

    def index_note(self, repository: MemoryRepository, note: MemoryNote, *, reconcile: bool = False) -> GraphRepository:
        node = self._node(note)
        if not reconcile and self.graph_repository.nodes.get(node.id) == node:
            return self.graph_repository
        if note.context.get("superseded"):
            node_id = f"node_{note.id}"
            for edge in tuple(self.graph_repository.edges.values()):
                if node_id in (edge.source_id, edge.target_id) and edge.key[3] in {"graph_builder", "memory_evolution"}:
                    self.graph_repository.remove_edge(edge)
        else:
            self._pairs(note, repository.list())
        self.graph_repository.add_node(node)
        return self.graph_repository
