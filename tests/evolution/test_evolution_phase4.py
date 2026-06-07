import unittest

from memory.evolution.memory_evolution_engine import MemoryEvolutionEngine
from memory.graph.graph_repository import GraphRepository
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType


class EvolutionPhase4Test(unittest.TestCase):
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
