"""Memory evolution result models."""

from __future__ import annotations

from dataclasses import dataclass

from memory.memory_note import MemoryNote


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    evolved_memories: tuple[MemoryNote, ...]
    source_memory_ids: tuple[str, ...]
    clusters: tuple[tuple[str, ...], ...]
