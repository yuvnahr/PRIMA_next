"""Workflow task routing."""

from __future__ import annotations

from dataclasses import dataclass

from workflow.execution_context import ExecutionContext
from workflow.workflow_state import WorkflowPhase


@dataclass(frozen=True, slots=True)
class TaskRoute:
    """Ordered route for a workflow execution."""

    phases: tuple[WorkflowPhase, ...]


class TaskRouter:
    """Determines the phase route for a user input."""

    DEFAULT_ROUTE = (
        WorkflowPhase.AFFECT,
        WorkflowPhase.MEMORY_RETRIEVAL,
        WorkflowPhase.PLANNING,
        WorkflowPhase.REFLECTION,
        WorkflowPhase.ACTION,
        WorkflowPhase.OUTPUT,
    )

    def route(self, context: ExecutionContext) -> TaskRoute:
        """Return the route for the current context.

        Future task-specific branching belongs here, keeping subsystem
        components unaware of each other.
        """
        requested = context.metadata.get("route")
        if requested:
            return TaskRoute(tuple(WorkflowPhase(phase) for phase in requested))
        return TaskRoute(self.DEFAULT_ROUTE)
