"""Workflow state models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WorkflowPhase(str, Enum):
    """Ordered cognitive workflow phases."""

    INPUT = "input"
    STATE_LOAD = "state_load"
    AFFECT = "affect"
    MEMORY_RETRIEVAL = "memory_retrieval"
    EVIDENCE_ACQUISITION = "evidence_acquisition"
    PLANNING = "planning"
    WORLD_SIMULATION = "world_simulation"
    UNCERTAINTY_ESTIMATION = "uncertainty_estimation"
    EXECUTION_DECISION = "execution_decision"
    REFLECTION = "reflection"
    ACTION = "action"
    ANSWER_GENERATION = "answer_generation"
    OUTPUT_VALIDATION = "output_validation"
    DOCUMENT_INGESTION = "document_ingestion"
    OUTPUT = "output"
    STATE_COMMIT = "state_commit"
    MEMORY_COMMIT = "memory_commit"
    MAINTENANCE_ENQUEUE = "maintenance_enqueue"


class WorkflowStatus(str, Enum):
    """Execution lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class WorkflowState:
    """Mutable execution state owned by the workflow layer."""

    status: WorkflowStatus = WorkflowStatus.PENDING
    current_phase: WorkflowPhase = WorkflowPhase.INPUT
    completed_phases: list[WorkflowPhase] = field(default_factory=list)
    retry_counts: dict[WorkflowPhase, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)

    def mark_phase_complete(self, phase: WorkflowPhase) -> None:
        """Record a completed phase without duplicating entries."""
        if phase not in self.completed_phases:
            self.completed_phases.append(phase)

    def increment_retry(self, phase: WorkflowPhase) -> int:
        """Increment and return the retry count for a phase."""
        self.retry_counts[phase] = self.retry_counts.get(phase, 0) + 1
        return self.retry_counts[phase]
