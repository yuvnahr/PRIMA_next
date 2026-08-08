"""PRIMA-NEXT Hierarchical Memory Fabric."""

from typing import Any

__all__ = [
    "InMemoryMemoryRepository",
    "MemoryLevel",
    "MemoryIndex",
    "MemoryNote",
    "MemoryRepository",
    "MemoryStore",
    "MemoryType",
    "format_memory_note",
]


def __getattr__(name: str) -> Any:
    if name in {"MemoryNote", "format_memory_note"}:
        from memory.memory_note import MemoryNote, format_memory_note

        return {"MemoryNote": MemoryNote, "format_memory_note": format_memory_note}[name]
    if name in {"InMemoryMemoryRepository", "MemoryRepository"}:
        from memory.memory_repository import InMemoryMemoryRepository, MemoryRepository

        return {"InMemoryMemoryRepository": InMemoryMemoryRepository, "MemoryRepository": MemoryRepository}[name]
    if name == "MemoryStore":
        from memory.memory_store import MemoryStore

        return MemoryStore
    if name == "MemoryIndex":
        from memory.memory_index import MemoryIndex

        return MemoryIndex
    if name in {"MemoryLevel", "MemoryType"}:
        from memory.memory_types import MemoryLevel, MemoryType

        return {"MemoryLevel": MemoryLevel, "MemoryType": MemoryType}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
