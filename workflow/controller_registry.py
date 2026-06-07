"""Controller protocols and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from workflow.execution_context import ExecutionContext
from workflow.workflow_state import WorkflowPhase


@runtime_checkable
class WorkflowController(Protocol):
    """Protocol implemented by workflow-managed subsystem controllers."""

    phase: WorkflowPhase

    async def execute(self, context: ExecutionContext) -> Any:
        """Execute one workflow phase."""


@dataclass(slots=True)
class ControllerRegistry:
    """Dependency-injected controller registry."""

    controllers: dict[WorkflowPhase, WorkflowController]

    def get(self, phase: WorkflowPhase) -> WorkflowController:
        """Return the controller for a workflow phase."""
        try:
            return self.controllers[phase]
        except KeyError as exc:
            raise KeyError(f"No controller registered for phase {phase.value}") from exc

    def register(self, controller: WorkflowController) -> None:
        """Register or replace a controller."""
        self.controllers[controller.phase] = controller
