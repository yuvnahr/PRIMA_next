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


def test_promotion_has_one_owner_and_is_retry_safe(tmp_path):
    import pytest

    from memory.memory_repository import ChromaMemoryRepository

    for repository in (InMemoryMemoryRepository(), ChromaMemoryRepository(str(tmp_path))):
        original = repository.add(MemoryNote.create("active context", memory_type=MemoryType.WORKING))
        (target,) = ConsolidationEngine(repository).run()
        assert repository.get(original.id, MemoryType.WORKING) is None
        assert [row.id for row in repository.list()] == [original.id]
        assert repository.move(original, target).id == target.id
        with pytest.raises(ValueError):
            repository.update(original)
        if isinstance(repository, ChromaMemoryRepository):
            assert ChromaMemoryRepository(str(tmp_path)).get(original.id).memory_type == MemoryType.EPISODIC


def test_interrupted_promotion_recovers_before_serving_reads(tmp_path, monkeypatch):
    import pytest

    from memory.memory_repository import ChromaMemoryRepository

    repository = ChromaMemoryRepository(str(tmp_path))
    source = repository.add(MemoryNote.create("pending work", memory_type=MemoryType.WORKING))
    collection = repository.collections[MemoryType.WORKING]
    with monkeypatch.context() as fault:

        def fail_delete(*args, **kwargs):
            raise OSError("interrupted after verified copy")

        fault.setattr(collection, "delete", fail_delete)
        with pytest.raises(OSError, match="after verified copy"):
            ConsolidationEngine(repository).run()
        with pytest.raises(OSError):
            repository.list()
    reopened = ChromaMemoryRepository(str(tmp_path))
    (row,) = reopened.list()
    assert row.id == source.id and row.memory_type == MemoryType.EPISODIC
    assert reopened.move(source, row).to_record() == row.to_record()
