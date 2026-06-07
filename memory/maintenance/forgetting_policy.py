"""Soft forgetting policy."""

from __future__ import annotations

from dataclasses import dataclass

from memory.memory_note import MemoryNote


@dataclass(frozen=True, slots=True)
class ForgettingPolicy:
    archive_threshold: float = 0.15

    def apply(self, note: MemoryNote, decayed_retention: float) -> MemoryNote:
        metadata = dict(note.retrieval_metadata)
        metadata["retrieval_suppressed"] = decayed_retention < self.archive_threshold
        metadata["archive_status"] = "archived" if decayed_retention < self.archive_threshold else "active"
        return note.with_updates(retention_score=decayed_retention, retrieval_metadata=metadata)
