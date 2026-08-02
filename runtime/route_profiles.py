"""Deterministic task/profile route plans for the canonical runtime boundary."""

from __future__ import annotations

from dataclasses import dataclass

from runtime.contracts import ExecutionProfile, RuntimeComponent, TaskKind


class InvalidRouteError(ValueError):
    """Raised when a task/profile combination has no valid route."""


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """Ordered component plan selected before workflow execution."""

    task_kind: TaskKind
    profile: ExecutionProfile
    components: tuple[RuntimeComponent, ...]

    @property
    def name(self) -> str:
        """Return the stable route identifier."""

        return f"{self.task_kind.value}:{self.profile.value}"

    @property
    def skipped_components(self) -> tuple[RuntimeComponent, ...]:
        """Return every known component not selected by this route."""

        return tuple(component for component in RuntimeComponent if component not in self.components)


CONTROL = (
    RuntimeComponent.INPUT_PARSER,
    RuntimeComponent.TASK_SCHEDULER,
    RuntimeComponent.STATE_MANAGER,
    RuntimeComponent.POLICY_ROUTER,
    RuntimeComponent.EXECUTION_DISPATCHER,
    RuntimeComponent.PERSISTENT_STATE,
)
RETRIEVAL = (
    RuntimeComponent.QUERY_REWRITER,
    RuntimeComponent.DENSE_RETRIEVAL,
    RuntimeComponent.SPARSE_RETRIEVAL,
    RuntimeComponent.TEMPORAL_RETRIEVAL,
    RuntimeComponent.FUSION,
    RuntimeComponent.RERANKER,
    RuntimeComponent.CONFIDENCE_ESTIMATOR,
    RuntimeComponent.CONTEXT_COMPRESSOR,
)
COMMIT = (
    RuntimeComponent.OUTPUT_VALIDATOR,
    RuntimeComponent.OUTPUT_SHAPER,
    RuntimeComponent.STATE_COMMIT,
    RuntimeComponent.MEMORY_COMMIT,
    RuntimeComponent.MAINTENANCE_EVENTS,
)
MODEL_ONLY = CONTROL + (RuntimeComponent.MODEL_EXECUTOR,) + COMMIT
SIMPLE_RAG = CONTROL + RETRIEVAL + (RuntimeComponent.MODEL_EXECUTOR,) + COMMIT
PRIMA_FULL = CONTROL + (
    RuntimeComponent.AFFECT_ENGINE,
) + RETRIEVAL[:4] + (
    RuntimeComponent.GRAPH_TRAVERSAL,
    RuntimeComponent.GRAPH_REASONING,
) + RETRIEVAL[4:] + (
    RuntimeComponent.PLANNER,
    RuntimeComponent.WORLD_MODEL,
    RuntimeComponent.UNCERTAINTY_ESTIMATOR,
    RuntimeComponent.REFLECTION,
    RuntimeComponent.MODEL_EXECUTOR,
) + COMMIT
INGESTION_ONLY = CONTROL[:5] + (
    RuntimeComponent.DOCUMENT_ENCODER,
    RuntimeComponent.MEMORY_INDEX,
    RuntimeComponent.MEMORY_COMMIT,
    RuntimeComponent.STATE_COMMIT,
    RuntimeComponent.MAINTENANCE_EVENTS,
    RuntimeComponent.OUTPUT_SHAPER,
)
AFFECT_ONLY = CONTROL + (
    RuntimeComponent.EMOTION_CLASSIFIER,
    RuntimeComponent.AFFECT_ENGINE,
    RuntimeComponent.STATE_COMMIT,
    RuntimeComponent.OUTPUT_SHAPER,
)
TOOL_FULL = CONTROL + (
    RuntimeComponent.AFFECT_ENGINE,
    RuntimeComponent.PLANNER,
    RuntimeComponent.WORLD_MODEL,
    RuntimeComponent.UNCERTAINTY_ESTIMATOR,
    RuntimeComponent.REFLECTION,
    RuntimeComponent.TOOL_EXECUTOR,
) + COMMIT


def _plan(task_kind: TaskKind, profile: ExecutionProfile, components: tuple[RuntimeComponent, ...]) -> RoutePlan:
    return RoutePlan(task_kind=task_kind, profile=profile, components=components)


_ROUTES = {
    (TaskKind.CONVERSATION, ExecutionProfile.MODEL_ONLY): _plan(TaskKind.CONVERSATION, ExecutionProfile.MODEL_ONLY, MODEL_ONLY),
    (TaskKind.CONVERSATION, ExecutionProfile.SIMPLE_RAG): _plan(TaskKind.CONVERSATION, ExecutionProfile.SIMPLE_RAG, SIMPLE_RAG),
    (TaskKind.CONVERSATION, ExecutionProfile.PRIMA_FULL): _plan(TaskKind.CONVERSATION, ExecutionProfile.PRIMA_FULL, PRIMA_FULL),
    (TaskKind.FACTUAL_QA, ExecutionProfile.MODEL_ONLY): _plan(TaskKind.FACTUAL_QA, ExecutionProfile.MODEL_ONLY, MODEL_ONLY),
    (TaskKind.FACTUAL_QA, ExecutionProfile.SIMPLE_RAG): _plan(TaskKind.FACTUAL_QA, ExecutionProfile.SIMPLE_RAG, SIMPLE_RAG),
    (TaskKind.FACTUAL_QA, ExecutionProfile.PRIMA_FULL): _plan(TaskKind.FACTUAL_QA, ExecutionProfile.PRIMA_FULL, PRIMA_FULL),
    (TaskKind.DOCUMENT_INGESTION, ExecutionProfile.INGESTION_ONLY): _plan(TaskKind.DOCUMENT_INGESTION, ExecutionProfile.INGESTION_ONLY, INGESTION_ONLY),
    (TaskKind.EMOTION_CLASSIFICATION, ExecutionProfile.AFFECT_ONLY): _plan(TaskKind.EMOTION_CLASSIFICATION, ExecutionProfile.AFFECT_ONLY, AFFECT_ONLY),
    (TaskKind.TOOL_REQUEST, ExecutionProfile.PRIMA_FULL): _plan(TaskKind.TOOL_REQUEST, ExecutionProfile.PRIMA_FULL, TOOL_FULL),
}


def route_matrix() -> dict[tuple[TaskKind, ExecutionProfile], RoutePlan | None]:
    """Return a deterministic decision for every task/profile combination."""

    return {
        (task_kind, profile): _ROUTES.get((task_kind, profile))
        for task_kind in TaskKind
        for profile in ExecutionProfile
    }


def select_route(task_kind: TaskKind, profile: ExecutionProfile) -> RoutePlan:
    """Return a valid route or raise a clear configuration error."""

    route = _ROUTES.get((task_kind, profile))
    if route is not None:
        return route
    allowed = ", ".join(
        candidate.value
        for candidate in ExecutionProfile
        if (task_kind, candidate) in _ROUTES
    )
    raise InvalidRouteError(
        f"Execution profile '{profile.value}' is invalid for task '{task_kind.value}'; allowed profiles: {allowed}."
    )
