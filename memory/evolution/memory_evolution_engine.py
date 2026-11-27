"""Memory evolution pipeline."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict

from memory.evolution.evolution_policy import EvolutionPolicy
from memory.evolution.evolution_result import EvolutionResult
from memory.evolution.lineage_tracker import LineageTracker
from memory.evolution.semantic_abstraction_engine import SemanticAbstractionEngine
from memory.graph.graph_builder import GraphBuilder
from memory.graph.graph_edge import GraphEdge
from memory.graph.graph_node import GraphNode
from memory.graph.graph_reasoning_engine import GraphReasoningEngine
from memory.graph.graph_repository import GraphRepository
from memory.memory_index import MemoryIndex
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
        with self.repository.transaction():
            return self._evolve(source_type)

    def _evolve(self, source_type: MemoryType) -> EvolutionResult:
        GraphBuilder(self.graph_repository).rebuild_for_type(self.repository, source_type)
        sources = {
            f"node_{note.id}": note
            for note in self.repository.list(source_type)
            if not note.lineage.parent_ids
            and note.memory_level is not MemoryLevel.SEMANTIC_ABSTRACTION
            and note.context.get("source") != "memory_evolution"
            and note.retrieval_metadata.get("archive_status") != "archived"
            and not note.context.get("superseded")
            and note.salience_score >= self.policy.min_salience_for_evolution
        }
        scopes: dict[tuple[str, str], set[str]] = defaultdict(set)
        for node_id, note in sources.items():
            scope = (
                str(note.context.get("source_conversation_id") or note.context.get("conversation_id", "")),
                str(note.context.get("user_id", "")),
            )
            if not scope[0]:
                scope = str(note.context.get("session_id", "")), scope[1]
            scopes[scope].add(node_id)
        communities = [
            group
            for ids in scopes.values()
            for group in GraphReasoningEngine(self.graph_repository.induced(ids)).detect_communities()
        ]
        active = set()
        evolved: list[MemoryNote] = []
        clusters: list[tuple[str, ...]] = []

        for community in communities:
            parents = sorted((sources[key] for key in community if key in sources), key=lambda item: item.id)
            if len(parents) < self.policy.min_cluster_size:
                continue
            kind = type(self.abstraction_engine)
            version = (
                "1"
                if kind is SemanticAbstractionEngine
                else getattr(self.abstraction_engine, "derivation_version", None)
            )
            if not version:
                raise ValueError("Custom summarizer requires an explicit derivation_version.")
            inputs = (
                [(p.id, p.content, p.keywords, p.context) for p in parents]
                if kind is SemanticAbstractionEngine
                else [p.to_record() for p in parents]
            )
            identity = (source_type.value, inputs, asdict(self.policy), kind.__module__, kind.__qualname__, version)
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            summary_id = f"summary_{digest}"
            active.add(summary_id)
            storage = self.repository.repository if isinstance(self.repository, MemoryIndex) else self.repository
            existing = storage.get(summary_id, MemoryType.SEMANTIC)
            if existing:
                if existing.context.get("superseded"):
                    existing = self.repository.update(
                        existing.with_updates(context={**existing.context, "superseded": False})
                    )
                self._link(existing, parents)
                continue
            summary = self.abstraction_engine.summarize_cluster(parents, self.policy.max_summary_words)
            lineage = self.lineage_tracker.build_lineage(parents)
            evolved_note = MemoryNote.create(
                content=summary,
                note_id=summary_id,
                memory_type=MemoryType.SEMANTIC,
                memory_level=MemoryLevel.SEMANTIC_ABSTRACTION,
                context={
                    **{
                        key: parents[0].context[key]
                        for key in ("source_conversation_id", "conversation_id", "session_id", "user_id")
                        if key in parents[0].context
                    },
                    "source_type": source_type.value,
                    "type": "semantic_abstraction",
                    "source": "memory_evolution",
                    "references": list(lineage.parent_ids),
                    **(
                        {"summary_kind": "fact_free_cluster_description", "evidence_eligible": False}
                        if summary
                        == SemanticAbstractionEngine().summarize_cluster(parents, self.policy.max_summary_words)
                        else {}
                    ),
                },
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

            self._link(evolved_note, parents)

        for old in self.repository.list(MemoryType.SEMANTIC):
            if (
                old.context.get("source") == "memory_evolution"
                and old.id not in active
                and old.context.get("source_type", "episodic") == source_type.value
                and not old.context.get("superseded")
            ):
                self.repository.update(old.with_updates(context={**old.context, "superseded": True}))

        return EvolutionResult(
            evolved_memories=tuple(evolved),
            source_memory_ids=tuple(dict.fromkeys(memory_id for cluster in clusters for memory_id in cluster)),
            clusters=tuple(clusters),
        )

    def _link(self, note: MemoryNote, parents: list[MemoryNote]) -> None:
        node_id = f"node_{note.id}"
        self.graph_repository.add_node(
            GraphNode(node_id, note.id, (note.memory_type.value, note.memory_level.value), {"keywords": note.keywords})
        )
        for parent in parents:
            self.graph_repository.add_edge(
                GraphEdge(
                    f"node_{parent.id}",
                    node_id,
                    "evolves_into",
                    metadata={"generation": note.lineage.generation, "producer": "memory_evolution"},
                )
            )
