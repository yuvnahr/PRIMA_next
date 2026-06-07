import unittest

from memory.memory_lineage import MemoryLineage
from memory.memory_note import MemoryNote, format_memory_note
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType
from memory.stores.episodic_memory import EpisodicMemory
from memory.stores.working_memory import WorkingMemory


class MemoryPhase1Test(unittest.TestCase):
    def test_memory_note_creation_and_backward_formatter(self) -> None:
        note = MemoryNote.create("I baked sourdough bread for a surprise party")

        self.assertTrue(note.id.startswith("mem_"))
        self.assertEqual(note.memory_type, MemoryType.EPISODIC)
        self.assertIn("sourdough", " ".join(note.keywords))

        formatted = format_memory_note(
            {
                "raw_text": "I baked sourdough bread",
                "dominant_emotions": {"joy": 0.8},
                "vector_embedding": [0.1] * 64,
            }
        )
        self.assertIn("metadata", formatted)
        self.assertEqual(len(formatted["embedding"]), 64)

    def test_logical_stores_are_separated(self) -> None:
        repository = InMemoryMemoryRepository()
        working = WorkingMemory(repository)
        episodic = EpisodicMemory(repository)

        working_note = working.add("temporary active context")
        episodic_note = episodic.add("raw user event")

        self.assertEqual(working.get(working_note.id), working_note)
        self.assertIsNone(working.get(episodic_note.id))
        self.assertEqual(len(repository.list()), 2)

    def test_lineage_tracking(self) -> None:
        lineage = MemoryLineage(parent_ids=("a", "b"), origin_memory_ids=("a", "b"))
        next_lineage = lineage.next_generation(("theme_x",))

        self.assertEqual(next_lineage.generation, 1)
        self.assertIn("theme_x", next_lineage.ancestor_ids)


if __name__ == "__main__":
    unittest.main()
