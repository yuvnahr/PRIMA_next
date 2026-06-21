"""Tests for novelty scoring."""

from __future__ import annotations

from memory.maintenance.memory_importance import MemoryImportanceEngine
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository


def test_repeated_memory_has_low_novelty() -> None:
    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("Please remember my allergy to peanuts."))

    novelty = MemoryImportanceEngine(repository).novelty_score("Please remember my allergy to peanuts.")

    assert novelty < 0.05


def test_new_memory_has_high_novelty() -> None:
    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("Please remember my allergy to peanuts."))

    novelty = MemoryImportanceEngine(repository).novelty_score("My new project is about adaptive retrieval benchmarks.")

    assert novelty > 0.3
