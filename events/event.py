"""Structured events for asynchronous PRIMA-NEXT communication."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from events.event_types import EventType


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable event envelope published through the in-process event bus."""

    event_type: EventType
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"event_{uuid.uuid4()}")
    topic: str = "default"
    execution_id: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", EventType(self.event_type))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "topic", str(self.topic))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def create(
        cls,
        event_type: EventType,
        source: str,
        payload: dict[str, Any] | None = None,
        *,
        topic: str = "default",
        execution_id: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """Create an event with default identifiers and UTC timestamp."""
        return cls(
            event_type=event_type,
            source=source,
            payload=payload or {},
            topic=topic,
            execution_id=execution_id,
            correlation_id=correlation_id,
            causation_id=causation_id,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event into plain Python values."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "topic": self.topic,
            "execution_id": self.execution_id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp.isoformat(),
        }
