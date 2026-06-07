import unittest

from memory.maintenance.consolidation_engine import ConsolidationEngine
from memory.maintenance.forgetting_policy import ForgettingPolicy
from memory.maintenance.salience_manager import SalienceManager
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType


class MaintenancePhase5Test(unittest.TestCase):
    def test_salience_and_soft_forgetting(self) -> None:
        salience = SalienceManager().score(
            emotion_intensity=0.8,
            retrieval_frequency=0.5,
            novelty=0.5,
            state_relevance=0.7,
        )
        self.assertGreater(salience, 0.5)

        note = MemoryNote.create("low retention memory", retention_score=0.1)
        forgotten = ForgettingPolicy(archive_threshold=0.2).apply(note, decayed_retention=0.1)
        self.assertEqual(forgotten.retrieval_metadata["archive_status"], "archived")

    def test_consolidation_promotes_working_to_episodic(self) -> None:
        repository = InMemoryMemoryRepository()
        note = repository.add(
            MemoryNote.create("active context", memory_type=MemoryType.WORKING, retention_score=0.9)
        )

        updated = ConsolidationEngine(repository).run()

        self.assertTrue(updated)
        self.assertIsNotNone(repository.get(note.id, MemoryType.EPISODIC))


if __name__ == "__main__":
    unittest.main()
