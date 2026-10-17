"""In-memory event store for local PRIMA-NEXT event history."""

from __future__ import annotations

from dataclasses import dataclass, field

from events.event import Event
from events.event_types import EventType


@dataclass(slots=True)
class EventStore:
    """Append-only in-memory event store."""

    _events: list[Event] = field(default_factory=list)
    _by_execution: dict[str | None, list[Event]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for event in self._events:
            self._by_execution.setdefault(event.execution_id, []).append(event)

    def append(self, event: Event) -> None:
        """Append one event to the store."""
        self._events.append(event)
        self._by_execution.setdefault(event.execution_id, []).append(event)

    def all(self) -> tuple[Event, ...]:
        """Return all stored events in insertion order."""
        return tuple(self._events)

    def filter(
        self,
        *,
        event_type: EventType | None = None,
        topic: str | None = None,
        execution_id: str | None = None,
        source: str | None = None,
    ) -> tuple[Event, ...]:
        """Return events matching optional filters."""
        events = self._events if execution_id is None else self._by_execution.get(execution_id, [])
        if event_type is not None:
            events = [event for event in events if event.event_type == EventType(event_type)]
        if topic is not None:
            events = [event for event in events if event.topic == topic]
        if source is not None:
            events = [event for event in events if event.source == source]
        return tuple(events)

    def clear(self) -> None:
        """Remove all stored events."""
        self._events.clear()
        self._by_execution.clear()
