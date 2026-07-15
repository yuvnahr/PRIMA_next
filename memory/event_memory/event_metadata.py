"""Structured metadata for event-level memories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EventMetadata:
    event_type: str = "conversation_event"
    participants: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    locations: tuple[str, ...] = ()
    temporal_expressions: tuple[str, ...] = ()
    relationships: tuple[str, ...] = ()
    preferences: tuple[str, ...] = ()
    emotion: str | None = None
    importance: float = 0.0
    source_turn_range: tuple[int, int] = (0, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "participants": list(self.participants),
            "objects": list(self.objects),
            "locations": list(self.locations),
            "temporal_expressions": list(self.temporal_expressions),
            "relationships": list(self.relationships),
            "preferences": list(self.preferences),
            "emotion": self.emotion,
            "importance": self.importance,
            "source_turn_range": list(self.source_turn_range),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> EventMetadata:
        if not isinstance(payload, dict):
            return cls()
        source_range = payload.get("source_turn_range") or (0, 0)
        if not isinstance(source_range, list | tuple) or len(source_range) != 2:
            source_range = (0, 0)
        return cls(
            event_type=str(payload.get("event_type") or "conversation_event"),
            participants=tuple(str(item) for item in payload.get("participants", ()) or ()),
            objects=tuple(str(item) for item in payload.get("objects", ()) or ()),
            locations=tuple(str(item) for item in payload.get("locations", ()) or ()),
            temporal_expressions=tuple(str(item) for item in payload.get("temporal_expressions", ()) or ()),
            relationships=tuple(str(item) for item in payload.get("relationships", ()) or ()),
            preferences=tuple(str(item) for item in payload.get("preferences", ()) or ()),
            emotion=str(payload["emotion"]) if payload.get("emotion") else None,
            importance=float(payload.get("importance", 0.0) or 0.0),
            source_turn_range=(int(source_range[0]), int(source_range[1])),
        )


def unique_sorted(values: Any) -> tuple[str, ...]:
    cleaned = {str(value).strip() for value in values if str(value).strip()}
    return tuple(sorted(cleaned, key=lambda item: item.lower()))
