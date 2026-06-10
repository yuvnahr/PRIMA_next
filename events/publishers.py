"""Publisher helpers for common PRIMA-NEXT events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from events.event import Event
from events.event_types import EventType


@dataclass(frozen=True, slots=True)
class EventPublisher:
    """Factory for typed events from one logical source."""

    source: str
    topic: str = "default"

    def memory_created(self, memory_id: str, payload: dict[str, Any] | None = None) -> Event:
        """Create a memory-created event."""
        return self._event(EventType.MEMORY_CREATED, {"memory_id": memory_id, **(payload or {})})

    def reflection_triggered(self, reason: str, payload: dict[str, Any] | None = None) -> Event:
        """Create a reflection-triggered event."""
        return self._event(EventType.REFLECTION_TRIGGERED, {"reason": reason, **(payload or {})})

    def state_changed(self, state_name: str, payload: dict[str, Any] | None = None) -> Event:
        """Create a state-changed event."""
        return self._event(EventType.STATE_CHANGED, {"state_name": state_name, **(payload or {})})

    def plan_failed(self, plan_id: str, reason: str, payload: dict[str, Any] | None = None) -> Event:
        """Create a plan-failed event."""
        return self._event(EventType.PLAN_FAILED, {"plan_id": plan_id, "reason": reason, **(payload or {})})

    def tool_executed(self, tool_name: str, status: str, payload: dict[str, Any] | None = None) -> Event:
        """Create a tool-executed event."""
        return self._event(EventType.TOOL_EXECUTED, {"tool_name": tool_name, "status": status, **(payload or {})})

    def custom(self, name: str, payload: dict[str, Any] | None = None) -> Event:
        """Create a custom event using the payload to hold the custom name."""
        return self._event(EventType.CUSTOM, {"name": name, **(payload or {})})

    def _event(self, event_type: EventType, payload: dict[str, Any]) -> Event:
        return Event.create(event_type=event_type, source=self.source, payload=payload, topic=self.topic)
