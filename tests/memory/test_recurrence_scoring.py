"""Tests for recurrence scoring."""

from __future__ import annotations

from memory.maintenance.memory_importance import MemoryImportanceEngine
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository


def test_recurrence_increases_for_repeated_topics() -> None:
    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("I am studying robotics for my research project."))
    repository.add(MemoryNote.create("The robotics project needs better retrieval evaluation."))
    repository.add(MemoryNote.create("My research notes mention robotics and planning."))

    recurrence = MemoryImportanceEngine(repository).recurrence_score("Robotics research is my main project.")

    assert recurrence >= 0.4


def test_recurrence_is_low_for_single_mention() -> None:
    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("I am studying robotics for my research project."))

    recurrence = MemoryImportanceEngine(repository).recurrence_score("I bought basil for dinner.")

    assert recurrence == 0.0
