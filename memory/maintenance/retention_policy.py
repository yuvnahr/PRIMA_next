"""Retention policy."""

from __future__ import annotations

from dataclasses import dataclass

from memory.memory_note import MemoryNote


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    working_to_episodic_threshold: float = 0.2
    episodic_to_semantic_threshold: float = 0.65
    decay_rate: float = 0.96

    def should_promote_working(self, note: MemoryNote) -> bool:
        return note.retention_score >= self.working_to_episodic_threshold

    def should_consider_semantic(self, note: MemoryNote) -> bool:
        return note.salience_score >= self.episodic_to_semantic_threshold

    def decay(self, note: MemoryNote) -> float:
        return round(max(0.0, min(1.0, note.retention_score * self.decay_rate)), 6)
