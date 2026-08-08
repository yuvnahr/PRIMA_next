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

    ROUTES = {
        ("conversation", "model_only"): (
            WorkflowPhase.STATE_LOAD,
            WorkflowPhase.ANSWER_GENERATION,
            WorkflowPhase.OUTPUT_VALIDATION,
            WorkflowPhase.OUTPUT,
            WorkflowPhase.STATE_COMMIT,
            WorkflowPhase.MEMORY_COMMIT,
        ),
        ("conversation", "simple_rag"): (
            WorkflowPhase.STATE_LOAD,
            WorkflowPhase.EVIDENCE_ACQUISITION,
            WorkflowPhase.ANSWER_GENERATION,
            WorkflowPhase.OUTPUT_VALIDATION,
            WorkflowPhase.OUTPUT,
            WorkflowPhase.STATE_COMMIT,
            WorkflowPhase.MEMORY_COMMIT,
        ),
        ("conversation", "prima_full"): (
            WorkflowPhase.STATE_LOAD,
            WorkflowPhase.AFFECT,
            WorkflowPhase.EVIDENCE_ACQUISITION,
            WorkflowPhase.PLANNING,
            WorkflowPhase.WORLD_SIMULATION,
            WorkflowPhase.UNCERTAINTY_ESTIMATION,
            WorkflowPhase.EXECUTION_DECISION,
            WorkflowPhase.REFLECTION,
            WorkflowPhase.ACTION,
            WorkflowPhase.ANSWER_GENERATION,
            WorkflowPhase.OUTPUT_VALIDATION,
            WorkflowPhase.OUTPUT,
            WorkflowPhase.STATE_COMMIT,
            WorkflowPhase.MEMORY_COMMIT,
        ),
        ("factual_qa", "model_only"): (
            WorkflowPhase.STATE_LOAD,
            WorkflowPhase.ANSWER_GENERATION,
            WorkflowPhase.OUTPUT_VALIDATION,
            WorkflowPhase.OUTPUT,
            WorkflowPhase.STATE_COMMIT,
        ),
        ("factual_qa", "simple_rag"): (
            WorkflowPhase.STATE_LOAD,
            WorkflowPhase.EVIDENCE_ACQUISITION,
            WorkflowPhase.ANSWER_GENERATION,
            WorkflowPhase.OUTPUT_VALIDATION,
            WorkflowPhase.OUTPUT,
            WorkflowPhase.STATE_COMMIT,
        ),
        ("factual_qa", "prima_full"): (
            WorkflowPhase.STATE_LOAD,
            WorkflowPhase.AFFECT,
            WorkflowPhase.EVIDENCE_ACQUISITION,
            WorkflowPhase.PLANNING,
            WorkflowPhase.WORLD_SIMULATION,
            WorkflowPhase.UNCERTAINTY_ESTIMATION,
            WorkflowPhase.EXECUTION_DECISION,
            WorkflowPhase.REFLECTION,
            WorkflowPhase.ACTION,
            WorkflowPhase.ANSWER_GENERATION,
            WorkflowPhase.OUTPUT_VALIDATION,
            WorkflowPhase.OUTPUT,
            WorkflowPhase.STATE_COMMIT,
        ),
        ("document_ingestion", "ingestion_only"): (
            WorkflowPhase.STATE_LOAD,
            WorkflowPhase.DOCUMENT_INGESTION,
            WorkflowPhase.OUTPUT,
            WorkflowPhase.STATE_COMMIT,
        ),
        ("emotion_classification", "affect_only"): (
            WorkflowPhase.STATE_LOAD,
            WorkflowPhase.AFFECT,
            WorkflowPhase.OUTPUT,
            WorkflowPhase.STATE_COMMIT,
        ),
        ("tool_request", "prima_full"): (
            WorkflowPhase.STATE_LOAD,
            WorkflowPhase.AFFECT,
            WorkflowPhase.MEMORY_RETRIEVAL,
            WorkflowPhase.PLANNING,
            WorkflowPhase.WORLD_SIMULATION,
            WorkflowPhase.UNCERTAINTY_ESTIMATION,
            WorkflowPhase.EXECUTION_DECISION,
            WorkflowPhase.REFLECTION,
            WorkflowPhase.ACTION,
            WorkflowPhase.OUTPUT_VALIDATION,
            WorkflowPhase.OUTPUT,
            WorkflowPhase.STATE_COMMIT,
            WorkflowPhase.MEMORY_COMMIT,
        ),
    }

    def route(self, context: ExecutionContext) -> TaskRoute:
        """Return the route for the current context.

        Subsystem components remain unaware of task/profile selection.
        """
        requested = context.metadata.get("route")
        if requested:
            return TaskRoute(tuple(WorkflowPhase(phase) for phase in requested))
        key = (str(context.metadata.get("task_kind", "")), str(context.metadata.get("profile", "")))
        if key in self.ROUTES:
            return TaskRoute(self.ROUTES[key])
        return TaskRoute(self.DEFAULT_ROUTE)
