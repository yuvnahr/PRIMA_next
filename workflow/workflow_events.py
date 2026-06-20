"""Workflow event compatibility adapters over the shared event bus."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from events import Event, EventBus, EventType, PublishResult
from workflow.workflow_state import WorkflowPhase, WorkflowStatus


class WorkflowEventType(str, Enum):
    """Event names published by the cognitive control bus."""

    WORKFLOW_STARTED = "workflow_started"
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    PHASE_RETRY = "phase_retry"
    PHASE_FAILED = "phase_failed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    WORKFLOW_FAILED = "workflow_failed"


WORKFLOW_EVENT_TYPE_MAP = {
    WorkflowEventType.WORKFLOW_STARTED: EventType.WORKFLOW_STARTED,
    WorkflowEventType.PHASE_STARTED: EventType.WORKFLOW_PHASE_STARTED,
    WorkflowEventType.PHASE_COMPLETED: EventType.WORKFLOW_PHASE_COMPLETED,
    WorkflowEventType.PHASE_RETRY: EventType.WORKFLOW_PHASE_RETRY,
    WorkflowEventType.PHASE_FAILED: EventType.WORKFLOW_PHASE_FAILED,
    WorkflowEventType.WORKFLOW_COMPLETED: EventType.WORKFLOW_COMPLETED,
    WorkflowEventType.WORKFLOW_CANCELLED: EventType.WORKFLOW_CANCELLED,
    WorkflowEventType.WORKFLOW_FAILED: EventType.WORKFLOW_FAILED,
}


@dataclass(frozen=True, slots=True)
class WorkflowEvent:
    """Structured workflow event."""

    event_type: WorkflowEventType
    execution_id: str
    status: WorkflowStatus
    phase: WorkflowPhase | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", WorkflowEventType(self.event_type))
        object.__setattr__(self, "status", WorkflowStatus(self.status))
        if self.phase is not None:
            object.__setattr__(self, "phase", WorkflowPhase(self.phase))
        object.__setattr__(self, "payload", dict(self.payload))

    def to_event(self) -> Event:
        """Convert this workflow event into a shared bus event."""
        payload = dict(self.payload)
        if self.phase is not None:
            payload["phase"] = self.phase.value
        payload["workflow_status"] = self.status.value
        return Event(
            event_type=WORKFLOW_EVENT_TYPE_MAP[self.event_type],
            source="workflow",
            payload=payload,
            topic="workflow",
            execution_id=self.execution_id,
            metadata={"workflow_event_type": self.event_type.value},
            timestamp=self.timestamp,
        )

    @classmethod
    def from_event(cls, event: Event) -> WorkflowEvent:
        """Convert a shared bus event back into a workflow event."""
        workflow_type = event.metadata.get("workflow_event_type")
        if workflow_type is None:
            workflow_type = _workflow_type_from_event_type(event.event_type).value
        phase_value = event.payload.get("phase")
        return cls(
            event_type=WorkflowEventType(str(workflow_type)),
            execution_id=event.execution_id or "",
            status=WorkflowStatus(str(event.payload.get("workflow_status", WorkflowStatus.PENDING.value))),
            phase=WorkflowPhase(str(phase_value)) if phase_value else None,
            payload={key: value for key, value in event.payload.items() if key not in {"phase", "workflow_status"}},
            timestamp=event.timestamp,
        )


WorkflowEventHandler = Callable[[WorkflowEvent], None | Awaitable[None]]


class WorkflowEventBus:
    """Backward-compatible workflow event bus backed by the shared event bus."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self.event_bus = event_bus or EventBus()
        self._handlers: list[WorkflowEventHandler] = []

    @property
    def events(self) -> list[WorkflowEvent]:
        """Return workflow events published through this bus."""
        return [
            WorkflowEvent.from_event(event)
            for event in self.event_bus.store.filter(topic="workflow")
            if event.event_type in WORKFLOW_EVENT_TYPE_MAP.values()
        ]

    def subscribe(self, handler: WorkflowEventHandler) -> None:
        """Register a workflow event handler."""
        self._handlers.append(handler)

    async def publish(self, event: WorkflowEvent) -> PublishResult:
        """Publish a workflow event to the shared event bus and local handlers."""
        shared_event = event.to_event()

        async def dispatch_to_local_handlers(published_event: Event) -> None:
            workflow_event = WorkflowEvent.from_event(published_event)
            for handler in list(self._handlers):
                result = handler(workflow_event)
                if inspect.isawaitable(result):
                    await result

        transient = self.event_bus.subscribe(
            dispatch_to_local_handlers,
            event_types=(shared_event.event_type,),
            topics=("workflow",),
        )
        try:
            return await self.event_bus.publish(shared_event)
        finally:
            self.event_bus.unsubscribe(transient)


def _workflow_type_from_event_type(event_type: EventType) -> WorkflowEventType:
    for workflow_type, mapped_type in WORKFLOW_EVENT_TYPE_MAP.items():
        if mapped_type == event_type:
            return workflow_type
    raise ValueError(f"Event type '{event_type.value}' is not a workflow event.")
