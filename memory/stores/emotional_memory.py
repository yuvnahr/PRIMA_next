from memory.memory_store import MemoryStore
from memory.memory_types import MemoryLevel, MemoryType
from memory.memory_repository import MemoryRepository
from typing import Any


class EmotionalMemory(MemoryStore):
    def __init__(self, repository: MemoryRepository | Any) -> None:
        super().__init__(repository, MemoryType.EMOTIONAL, MemoryLevel.EMOTIONAL_TRACE)
