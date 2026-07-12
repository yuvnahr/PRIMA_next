"""Memory evolution pipeline."""

from __future__ import annotations

from memory.evolution.evolution_policy import EvolutionPolicy
from memory.evolution.evolution_result import EvolutionResult
from memory.evolution.lineage_tracker import LineageTracker
from memory.evolution.semantic_abstraction_engine import SemanticAbstractionEngine
from memory.graph.graph_builder import GraphBuilder
from memory.graph.graph_edge import GraphEdge
from memory.graph.graph_node import GraphNode
from memory.graph.graph_reasoning_engine import GraphReasoningEngine
from memory.graph.graph_repository import GraphRepository
from memory.memory_note import MemoryNote
from memory.memory_repository import MemoryRepository
from memory.memory_types import MemoryLevel, MemoryType


class MemoryEvolutionEngine:
    def __init__(
        self,
        repository: MemoryRepository,
        graph_repository: GraphRepository,
        policy: EvolutionPolicy | None = None,
        abstraction_engine: SemanticAbstractionEngine | None = None,
        lineage_tracker: LineageTracker | None = None,
    ) -> None:
        self.repository = repository
        self.graph_repository = graph_repository
        self.policy = policy or EvolutionPolicy()
        self.abstraction_engine = abstraction_engine or SemanticAbstractionEngine()
        self.lineage_tracker = lineage_tracker or LineageTracker()

    def evolve(self, source_type: MemoryType = MemoryType.EPISODIC) -> EvolutionResult:
        GraphBuilder(self.graph_repository).rebuild_for_type(self.repository, source_type)
        communities = GraphReasoningEngine(self.graph_repository).detect_communities()
        evolved: list[MemoryNote] = []
        clusters: list[tuple[str, ...]] = []

        for community in communities:
            notes = [
                self.repository.get(self.graph_repository.nodes[node_id].memory_id)
                for node_id in community
            ]
            parents = [
                note
                for note in notes
                if note is not None and note.salience_score >= self.policy.min_salience_for_evolution
            ]
            if len(parents) < self.policy.min_cluster_size:
                continue

            summary = self.abstraction_engine.summarize_cluster(parents, self.policy.max_summary_words)
            lineage = self.lineage_tracker.build_lineage(parents)
            evolved_note = MemoryNote.create(
                content=summary,
                memory_type=MemoryType.SEMANTIC,
                memory_level=MemoryLevel.SEMANTIC_ABSTRACTION,

                context={"type": "semantic_abstraction", "source": "memory_evolution", "references": list(lineage.parent_ids)},
                salience_score=max(parent.salience_score for parent in parents),
                retention_score=max(parent.retention_score for parent in parents),
            ).with_updates(
                lineage=lineage,
                evolution_metadata={
                    "source_cluster_size": len(parents),
                    "source_memory_ids": list(lineage.parent_ids),
                    "concepts": self.abstraction_engine.extract_concepts(parents[0]),
                },
            )
            self.repository.add(evolved_note)
            evolved.append(evolved_note)
            clusters.append(tuple(parent.id for parent in parents))

            evolved_node_id = f"node_{evolved_note.id}"
            self.graph_repository.add_node(
                self.graph_repository.nodes.get(evolved_node_id)
                or GraphNode(
                    id=evolved_node_id,
                    memory_id=evolved_note.id,
                    labels=(MemoryType.SEMANTIC.value, MemoryLevel.SEMANTIC_ABSTRACTION.value),
                    metadata={"keywords": evolved_note.keywords},
                )
            )
            for parent in parents:
                parent_node = self.graph_repository.find_by_memory_id(parent.id)
                if parent_node:
                    self.graph_repository.add_edge(
                        GraphEdge(
                            source_id=parent_node.id,
                            target_id=evolved_node_id,
                            relation_type="evolves_into",
                            weight=1.0,
                            metadata={"generation": lineage.generation},
                        )
                    )

        return EvolutionResult(
            evolved_memories=tuple(evolved),
            source_memory_ids=tuple(dict.fromkeys(memory_id for cluster in clusters for memory_id in cluster)),
            clusters=tuple(clusters),
        )

