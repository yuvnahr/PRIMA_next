"""Typed event construction for the local memory-maintenance cold path."""

from __future__ import annotations

from typing import Any

from events.event import Event
from events.event_types import EventType

MAINTENANCE_EVENT_TYPES = frozenset(
    {
        EventType.MEMORY_ADMITTED,
        EventType.MEMORY_ENCODED,
        EventType.SALIENCE_SCORED,
        EventType.SHORT_TERM_BUFFERED,
        EventType.CONSOLIDATION_REQUESTED,
        EventType.CONSOLIDATION_COMPLETED,
        EventType.ABSTRACTION_REQUESTED,
        EventType.ABSTRACTION_COMPLETED,
        EventType.GRAPH_INDEX_UPDATED,
        EventType.DECAY_APPLIED,
        EventType.MAINTENANCE_FAILED,
    }
)


def maintenance_event(
    event_type: EventType,
    *,
    memory_id: str,
    source: str,
    execution_id: str | None = None,
    causation_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Event:
    """Create a validated, versioned maintenance event for one memory."""

    if event_type not in MAINTENANCE_EVENT_TYPES:
        raise ValueError(f"{event_type.value!r} is not a maintenance event type.")
    if not memory_id.strip():
        raise ValueError("maintenance events require a memory_id.")
    return Event.create(
        event_type,
        source,
        {"schema_version": "1.0", "memory_id": memory_id, **(payload or {})},
        topic="memory_maintenance",
        execution_id=execution_id,
        correlation_id=memory_id,
        causation_id=causation_id,
    )
