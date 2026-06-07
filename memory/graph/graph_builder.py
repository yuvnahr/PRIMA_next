"""Build graph links from memory context chains."""

from __future__ import annotations

from memory.graph.graph_edge import GraphEdge
from memory.graph.graph_node import GraphNode
from memory.graph.graph_repository import GraphRepository
from memory.memory_note import calculate_smart_overlap
from memory.memory_repository import MemoryRepository
from memory.memory_types import MemoryType


class GraphBuilder:
    def __init__(self, graph_repository: GraphRepository) -> None:
        self.graph_repository = graph_repository

    def rebuild_for_type(self, repository: MemoryRepository, memory_type: MemoryType) -> GraphRepository:
        notes = repository.list(memory_type)
        for note in notes:
            self.graph_repository.add_node(
                GraphNode(
                    id=f"node_{note.id}",
                    memory_id=note.id,
                    labels=(note.memory_type.value, note.memory_level.value),
                    metadata={"keywords": note.keywords},
                )
            )

        for left_index, left in enumerate(notes):
            for right in notes[left_index + 1 :]:
                overlap = calculate_smart_overlap(left.keywords, right.keywords)
                if overlap <= 0:
                    continue
                weight = min(1.0, 0.35 + overlap * 0.2)
                self.graph_repository.add_edge(
                    GraphEdge(
                        source_id=f"node_{left.id}",
                        target_id=f"node_{right.id}",
                        relation_type="shared_context",
                        weight=weight,
                        metadata={"overlap": overlap},
                    )
                )
        return self.graph_repository
