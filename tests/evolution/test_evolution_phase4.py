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


def test_unchanged_sources_reuse_summary_and_retraction_retires_it():
    repository = InMemoryMemoryRepository()
    for name in ("a", "b", "c"):
        repository.add(
            MemoryNote.create(
                "shared bread", note_id=name, context={"source_conversation_id": "first" if name != "c" else "other"}
            )
        )
    engine = MemoryEvolutionEngine(repository, GraphRepository())
    (summary,) = engine.evolve().evolved_memories
    assert summary.lineage.parent_ids == ("a", "b")
    assert not engine.evolve().evolved_memories
    repository.update(repository.get("a").with_updates(retention_score=0.4))
    assert not engine.evolve().evolved_memories
    repository.delete("a", MemoryType.EPISODIC)
    assert not engine.evolve().evolved_memories
    assert repository.get(summary.id).context["superseded"]


def test_evolution_reopen_edit_and_restore_original_revision(tmp_path):
    from memory.memory_index import MemoryIndex
    from memory.memory_repository import ChromaMemoryRepository

    def open_index():
        return MemoryIndex.for_repository(ChromaMemoryRepository(str(tmp_path)))

    index = open_index()
    first = index.add(MemoryNote.create("shared tea", note_id="a"))
    index.add(MemoryNote.create("shared cake", note_id="b"))

    def evolve(index):
        return MemoryEvolutionEngine(index, index.graph_repository).evolve().evolved_memories

    (original,) = evolve(index)
    index = open_index()
    assert not evolve(index)
    index.update(first.with_updates(content="shared coffee"))
    (changed,) = evolve(index)
    assert changed.id != original.id and index.get(original.id) is None
    index.update(first)
    evolve(index)
    assert [n.id for n in index.list(MemoryType.SEMANTIC)] == [original.id]
