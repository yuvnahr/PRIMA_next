from memory.memory_store import MemoryStore
from memory.memory_types import MemoryLevel, MemoryType


class WorkingMemory(MemoryStore):
    def __init__(self, repository):
        super().__init__(repository, MemoryType.WORKING, MemoryLevel.RAW)
