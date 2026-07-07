"""Event-level memory model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory.event_memory.event_metadata import EventMetadata
from memory.memory_note import MemoryNote, stable_embedding
from memory.memory_types import MemoryLevel, MemoryType


@dataclass(frozen=True, slots=True)
class EventMemory:
    event_id: str
    conversation_id: str
    summary: str
    turns_included: tuple[str, ...]
    metadata: EventMetadata
    embedding_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "conversation_id": self.conversation_id,
            "summary": self.summary,
            "turns_included": list(self.turns_included),
            "metadata": self.metadata.to_dict(),
            "embedding_text": self.embedding_text,
        }

    def to_memory_note(self) -> MemoryNote:
        retrieval_metadata = {
            "event_id": self.event_id,
            "event_metadata": self.metadata.to_dict(),
            "source_turn_ids": list(self.turns_included),
            "embedding_text": self.embedding_text,
        }
        return MemoryNote.create(
            content=self.summary,
            memory_type=MemoryType.EPISODIC,
            memory_level=MemoryLevel.EPISODIC_EVENT,
            embedding=stable_embedding(self.embedding_text),
            context={
                "type": "event_memory",
                "conversation_id": self.conversation_id,
                "source_turn_ids": list(self.turns_included),
            },
            salience_score=self.metadata.importance,
            note_id=self.event_id,
        ).with_updates(retrieval_metadata=retrieval_metadata)
