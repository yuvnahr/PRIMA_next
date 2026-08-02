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
            WorkflowPhase.ANSWER_GENERATION,
            WorkflowPhase.OUTPUT,
            WorkflowPhase.MEMORY_COMMIT,
        ),
        ("conversation", "simple_rag"): (
            WorkflowPhase.EVIDENCE_ACQUISITION,
            WorkflowPhase.ANSWER_GENERATION,
            WorkflowPhase.OUTPUT,
            WorkflowPhase.MEMORY_COMMIT,
        ),
        ("conversation", "prima_full"): (
            WorkflowPhase.AFFECT,
            WorkflowPhase.EVIDENCE_ACQUISITION,
            WorkflowPhase.PLANNING,
            WorkflowPhase.REFLECTION,
            WorkflowPhase.ACTION,
            WorkflowPhase.ANSWER_GENERATION,
            WorkflowPhase.OUTPUT,
            WorkflowPhase.MEMORY_COMMIT,
        ),
        ("factual_qa", "model_only"): (WorkflowPhase.ANSWER_GENERATION, WorkflowPhase.OUTPUT),
        ("factual_qa", "simple_rag"): (
            WorkflowPhase.EVIDENCE_ACQUISITION,
            WorkflowPhase.ANSWER_GENERATION,
            WorkflowPhase.OUTPUT,
        ),
        ("factual_qa", "prima_full"): (
            WorkflowPhase.AFFECT,
            WorkflowPhase.EVIDENCE_ACQUISITION,
            WorkflowPhase.PLANNING,
            WorkflowPhase.REFLECTION,
            WorkflowPhase.ACTION,
            WorkflowPhase.ANSWER_GENERATION,
            WorkflowPhase.OUTPUT,
        ),
        ("document_ingestion", "ingestion_only"): (
            WorkflowPhase.DOCUMENT_INGESTION,
            WorkflowPhase.OUTPUT,
        ),
        ("emotion_classification", "affect_only"): (WorkflowPhase.AFFECT, WorkflowPhase.OUTPUT),
        ("tool_request", "prima_full"): (
            WorkflowPhase.AFFECT,
            WorkflowPhase.PLANNING,
            WorkflowPhase.ACTION,
            WorkflowPhase.OUTPUT,
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
