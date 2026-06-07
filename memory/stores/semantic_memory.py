from memory.memory_store import MemoryStore
from memory.memory_types import MemoryLevel, MemoryType


class SemanticMemory(MemoryStore):
    def __init__(self, repository):
        super().__init__(repository, MemoryType.SEMANTIC, MemoryLevel.SEMANTIC_ABSTRACTION)
