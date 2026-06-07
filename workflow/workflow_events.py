"""Workflow events and async event bus."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable

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


@dataclass(frozen=True, slots=True)
class WorkflowEvent:
    """Structured workflow event."""

    event_type: WorkflowEventType
    execution_id: str
    status: WorkflowStatus
    phase: WorkflowPhase | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


EventHandler = Callable[[WorkflowEvent], None | Awaitable[None]]


class WorkflowEventBus:
    """Async-safe in-process event bus.

    The bus owns asynchronous workflow communication. Handlers may be sync or
    async callables; publishing awaits async handlers before returning.
    """

    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []
        self.events: list[WorkflowEvent] = []

    def subscribe(self, handler: EventHandler) -> None:
        """Register a workflow event handler."""
        self._handlers.append(handler)

    async def publish(self, event: WorkflowEvent) -> None:
        """Publish an event to all subscribers."""
        self.events.append(event)
        for handler in list(self._handlers):
            result = handler(event)
            if inspect.isawaitable(result):
                await result
