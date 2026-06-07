from memory.memory_store import MemoryStore
from memory.memory_types import MemoryLevel, MemoryType


class EpisodicMemory(MemoryStore):
    def __init__(self, repository):
        super().__init__(repository, MemoryType.EPISODIC, MemoryLevel.EPISODIC_EVENT)
