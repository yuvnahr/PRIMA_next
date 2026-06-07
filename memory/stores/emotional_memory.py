from memory.memory_store import MemoryStore
from memory.memory_types import MemoryLevel, MemoryType


class EmotionalMemory(MemoryStore):
    def __init__(self, repository):
        super().__init__(repository, MemoryType.EMOTIONAL, MemoryLevel.EMOTIONAL_TRACE)
