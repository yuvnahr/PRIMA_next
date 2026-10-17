import unittest
from unittest.mock import patch

from memory.evolution.memory_evolution_engine import MemoryEvolutionEngine
from memory.graph.graph_repository import GraphRepository
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType


class EvolutionPhase4Test(unittest.TestCase):
    def test_community_iteration_order_cannot_change_abstraction_or_lineage(self) -> None:
        def evolve(reverse: bool):
            repository = InMemoryMemoryRepository()
            for index, content in enumerate(("Alpha shared bread", "Beta shared starter", "Gamma shared baking")):
                repository.add(MemoryNote.create(content, memory_type=MemoryType.EPISODIC,
                                               note_id=f"parent-{index}", salience_score=0.8))
            nodes = [f"node_parent-{index}" for index in range(3)]
            if reverse:
                nodes.reverse()
            with patch("memory.evolution.memory_evolution_engine.GraphReasoningEngine.detect_communities",
                       return_value=[nodes]):
                return MemoryEvolutionEngine(repository, GraphRepository()).evolve().evolved_memories[0]

        first, second = evolve(False), evolve(True)
        self.assertEqual(first.content, second.content)
        self.assertEqual(first.lineage, second.lineage)
        self.assertEqual(first.evolution_metadata, second.evolution_metadata)
        self.assertEqual(first.lineage.parent_ids, ("parent-0", "parent-1", "parent-2"))

    def test_evolution_creates_semantic_memory_with_lineage(self) -> None:
        repository = InMemoryMemoryRepository()
        repository.add(MemoryNote.create("I baked sourdough bread", memory_type=MemoryType.EPISODIC, salience_score=0.8))
        repository.add(MemoryNote.create("The sourdough starter needs feeding", memory_type=MemoryType.EPISODIC, salience_score=0.8))

        graph = GraphRepository()
        result = MemoryEvolutionEngine(repository, graph).evolve(MemoryType.EPISODIC)

        self.assertEqual(len(result.evolved_memories), 1)
        evolved = result.evolved_memories[0]
        self.assertEqual(evolved.memory_type, MemoryType.SEMANTIC)
        self.assertEqual(evolved.lineage.generation, 1)
        self.assertEqual(len(evolved.lineage.parent_ids), 2)


if __name__ == "__main__":
    unittest.main()
