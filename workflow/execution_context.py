"""Execution context owned by the workflow layer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from state.cognitive_state import CognitiveState
from workflow.correction_loop import CorrectionAttempt
from workflow.workflow_state import WorkflowState


@dataclass(slots=True)
class ExecutionContext:
    """Per-request cognitive execution context."""

    user_input: str
    execution_id: str = field(default_factory=lambda: f"exec_{uuid.uuid4()}")
    cognitive_state: CognitiveState = field(default_factory=CognitiveState)
    workflow_state: WorkflowState = field(default_factory=WorkflowState)
    affect_update: Any | None = None
    retrieval_response: Any | None = None
    compressed_context: Any | None = None
    reasoning_result: Any | None = None
    plan: Any | None = None
    world_prediction: Any | None = None
    uncertainty: Any | None = None
    execution_decision: Any | None = None
    reflection_result: Any | None = None
    action_result: Any | None = None
    generation_result: Any | None = None
    output_validation: Any | None = None
    ingestion_result: Any | None = None
    memory_notes_created: tuple[Any, ...] = ()
    memory_admission: Any | None = None
    maintenance_result: dict[str, Any] = field(default_factory=dict)
    correction_attempts: list[CorrectionAttempt] = field(default_factory=list)
    output: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cancellation_requested: bool = False

    def request_cancel(self) -> None:
        """Request cooperative cancellation before the next phase starts."""
        self.cancellation_requested = True
        self.touch()

    def touch(self) -> None:
        """Update the context mutation timestamp."""
        self.updated_at = datetime.now(timezone.utc)
