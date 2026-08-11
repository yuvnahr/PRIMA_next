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

    def route(self, context: ExecutionContext) -> TaskRoute:
        """Return the route for the current context.

        Subsystem components remain unaware of task/profile selection.
        """
        requested = context.metadata.get("route")
        if not isinstance(requested, tuple) or not requested:
            raise ValueError("Workflow execution requires a trusted canonical route plan.")
        return TaskRoute(tuple(WorkflowPhase(phase) for phase in requested))
