from typing import Any

from memory.memory_repository import MemoryRepository
from memory.memory_store import MemoryStore
from memory.memory_types import MemoryLevel, MemoryType


class EmotionalMemory(MemoryStore):
    def __init__(self, repository: MemoryRepository | Any) -> None:
        super().__init__(repository, MemoryType.EMOTIONAL, MemoryLevel.EMOTIONAL_TRACE)
