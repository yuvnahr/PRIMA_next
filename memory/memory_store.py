"""Logical memory store wrapper."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, List

from memory.memory_note import MemoryNote
from memory.memory_repository import MemoryRepository
from memory.memory_types import MemoryLevel, MemoryType


class MemoryStore:
    def __init__(
        self,
        repository: MemoryRepository,
        memory_type: MemoryType,
        default_level: MemoryLevel,
    ) -> None:
        self.repository = repository
        self.memory_type = memory_type
        self.default_level = default_level

    def add(self, content: str, **kwargs: Any) -> MemoryNote:
        note = MemoryNote.create(
            content=content,
            memory_type=self.memory_type,
            memory_level=self.default_level,
            **kwargs,
        )
        return self.repository.add(note)

    def add_note(self, note: MemoryNote) -> MemoryNote:
        if note.memory_type != self.memory_type:
            note = note.with_updates(memory_type=self.memory_type, memory_level=self.default_level)
        return self.repository.add(note)

    def get(self, note_id: str) -> MemoryNote | None:
        return self.repository.get(note_id, self.memory_type)

    def list(self) -> List[MemoryNote]:
        return self.repository.list(self.memory_type)

    def query(self, embedding: Iterable[float], limit: int = 10) -> List[tuple[MemoryNote, float]]:
        return self.repository.query(embedding, self.memory_type, limit)
