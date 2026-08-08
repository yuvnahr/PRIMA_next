"""Cognitive Control Bus orchestration engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from reasoning.reflection_advisor import ReflectionAction, ReflectionAdvice
from state.cognitive_state import CognitiveState
from uncertainty.execution_gate import ExecutionDecision
from workflow.controller_registry import ControllerRegistry
from workflow.correction_loop import CorrectionLoop
from workflow.execution_context import ExecutionContext
from workflow.task_router import TaskRouter
from workflow.workflow_events import WorkflowEvent, WorkflowEventBus, WorkflowEventType
from workflow.workflow_state import WorkflowPhase, WorkflowStatus


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Workflow retry policy."""

    max_retries: int = 1


class WorkflowCancelledError(RuntimeError):
    """Raised when a workflow execution is cooperatively cancelled."""


class WorkflowExecutionError(RuntimeError):
    """Carry a failed execution context back to the canonical runtime."""

    def __init__(self, context: ExecutionContext) -> None:
        super().__init__("Workflow execution failed.")
        self.context = context


class OrchestrationEngine:
    """Owns the workflow execution lifecycle."""

    def __init__(
        self,
        registry: ControllerRegistry,
        router: TaskRouter | None = None,
        event_bus: WorkflowEventBus | None = None,
        retry_policy: RetryPolicy | None = None,
        correction_loop: CorrectionLoop | None = None,
    ) -> None:
        self.registry = registry
        self.router = router or TaskRouter()
        self.event_bus = event_bus or WorkflowEventBus()
        self.retry_policy = retry_policy or RetryPolicy()
        self.correction_loop = correction_loop or CorrectionLoop()

    async def execute(self, context: ExecutionContext) -> ExecutionContext:
        """Execute the routed cognitive workflow."""
        context.workflow_state.status = WorkflowStatus.RUNNING
        await self._publish(context, WorkflowEventType.WORKFLOW_STARTED)
        try:
            phases = list(self.router.route(context).phases)
            index = 0
            while index < len(phases):
                phase = phases[index]
                index += 1
                if self._skip_phase(context, phase):
                    continue
                await self._check_cancelled(context)
                context.workflow_state.current_phase = phase
                await self._publish(context, WorkflowEventType.PHASE_STARTED, phase)
                await self._execute_phase_with_retries(context, phase)
                context.workflow_state.mark_phase_complete(phase)
                await self._publish(context, WorkflowEventType.PHASE_COMPLETED, phase)
                if phase is WorkflowPhase.ANSWER_GENERATION:
                    context.metadata["generation_count"] = int(context.metadata.get("generation_count", 0)) + 1
                if phase is WorkflowPhase.EXECUTION_DECISION and self._requests_retrieval_retry(context):
                    context.metadata["retrieval_retry_count"] = int(
                        context.metadata.get("retrieval_retry_count", 0)
                    ) + 1
                    phases[index:index] = [
                        WorkflowPhase.EVIDENCE_ACQUISITION,
                        WorkflowPhase.PLANNING,
                        WorkflowPhase.WORLD_SIMULATION,
                        WorkflowPhase.UNCERTAINTY_ESTIMATION,
                        WorkflowPhase.EXECUTION_DECISION,
                    ]
                advice = self._phase_advice(context, phase)
                if advice is not None:
                    stages = self._correction_phases(context, advice, phase)
                    if stages:
                        phases[index:index] = stages
                if phase in {WorkflowPhase.OUTPUT_VALIDATION, WorkflowPhase.OUTPUT}:
                    self.correction_loop.complete(context)

            context.workflow_state.status = WorkflowStatus.COMPLETED
            await self._publish(context, WorkflowEventType.WORKFLOW_COMPLETED)
            return context
        except WorkflowCancelledError:
            context.workflow_state.status = WorkflowStatus.CANCELLED
            await self._publish(context, WorkflowEventType.WORKFLOW_CANCELLED, context.workflow_state.current_phase)
            return context
        except Exception as exc:
            context.workflow_state.status = WorkflowStatus.FAILED
            context.workflow_state.errors.append(str(exc))
            await self._publish(
                context,
                WorkflowEventType.WORKFLOW_FAILED,
                context.workflow_state.current_phase,
                {"error": str(exc)},
            )
            raise WorkflowExecutionError(context) from exc
        finally:
            context.touch()

    async def cancel(self, context: ExecutionContext) -> None:
        """Request cooperative cancellation for an execution context."""
        context.request_cancel()

    async def _execute_phase_with_retries(self, context: ExecutionContext, phase: WorkflowPhase) -> None:
        attempts = 0
        while True:
            try:
                controller = self.registry.get(phase)
                result = await controller.execute(context)
                self._route_phase_output(context, phase, result)
                context.touch()
                return
            except asyncio.CancelledError as exc:
                context.request_cancel()
                raise WorkflowCancelledError("Workflow task was cancelled.") from exc
            except Exception as exc:
                attempts += 1
                retry_count = context.workflow_state.increment_retry(phase)
                context.workflow_state.errors.append(f"{phase.value}: {exc}")
                if attempts > self.retry_policy.max_retries:
                    await self._publish(
                        context,
                        WorkflowEventType.PHASE_FAILED,
                        phase,
                        {"error": str(exc), "retry_count": retry_count},
                    )
                    raise
                await self._publish(
                    context,
                    WorkflowEventType.PHASE_RETRY,
                    phase,
                    {"error": str(exc), "retry_count": retry_count},
                )

    def _route_phase_output(self, context: ExecutionContext, phase: WorkflowPhase, result: object) -> None:
        if phase == WorkflowPhase.STATE_LOAD:
            if not isinstance(result, CognitiveState):
                raise TypeError("State load controller must return CognitiveState.")
            context.cognitive_state = result
        elif phase == WorkflowPhase.AFFECT:
            context.affect_update = result
        elif phase == WorkflowPhase.MEMORY_RETRIEVAL:
            context.retrieval_response = result
        elif phase == WorkflowPhase.EVIDENCE_ACQUISITION:
            context.reasoning_result = getattr(result, "answer_result", None)
            context.retrieval_response = getattr(result, "latest_retrieval", None)
        elif phase == WorkflowPhase.PLANNING:
            context.plan = result
        elif phase == WorkflowPhase.WORLD_SIMULATION:
            context.world_prediction = result
        elif phase == WorkflowPhase.UNCERTAINTY_ESTIMATION:
            context.uncertainty = result
        elif phase == WorkflowPhase.EXECUTION_DECISION:
            context.execution_decision = result
        elif phase == WorkflowPhase.REFLECTION:
            context.reflection_result = result
        elif phase == WorkflowPhase.ACTION:
            context.action_result = result
        elif phase == WorkflowPhase.ANSWER_GENERATION:
            context.generation_result = result
        elif phase == WorkflowPhase.OUTPUT_VALIDATION:
            context.output_validation = result
        elif phase == WorkflowPhase.DOCUMENT_INGESTION:
            context.ingestion_result = result
        elif phase == WorkflowPhase.OUTPUT:
            context.output = result
        elif phase == WorkflowPhase.STATE_COMMIT:
            if not isinstance(result, dict):
                raise TypeError("State commit controller must return a dictionary.")
            context.cognitive_state = result["state"]
            context.metadata["state_committed"] = bool(result["committed"])
        elif phase == WorkflowPhase.MEMORY_COMMIT:
            if not isinstance(result, dict):
                raise TypeError("Memory commit controller must return a dictionary.")
            context.memory_notes_created = tuple(result.get("notes", ()))
            context.memory_admission = result.get("admission")
        elif phase == WorkflowPhase.MAINTENANCE_ENQUEUE:
            if not isinstance(result, dict):
                raise TypeError("Maintenance enqueue controller must return a dictionary.")
            context.maintenance_result = result
        context.workflow_state.outputs[phase.value] = result

    def _requests_retrieval_retry(self, context: ExecutionContext) -> bool:
        decision = getattr(context.execution_decision, "decision", None)
        thresholds = getattr(context.execution_decision, "thresholds", {})
        maximum = min(
            int(thresholds.get("max_retrieval_retries", 0)),
            self.correction_loop.budget.max_retrieval_retries,
        )
        current = int(context.metadata.get("retrieval_retry_count", 0))
        return decision is ExecutionDecision.RETRY_RETRIEVAL and current < maximum

    def _skip_phase(self, context: ExecutionContext, phase: WorkflowPhase) -> bool:
        decision = getattr(context.execution_decision, "decision", None)
        terminal = str(context.metadata.get("correction_terminal_action", ""))
        if phase is WorkflowPhase.REFLECTION:
            return context.execution_decision is not None and decision is not ExecutionDecision.REFLECT
        if phase in {WorkflowPhase.ACTION, WorkflowPhase.ANSWER_GENERATION}:
            return terminal in {ReflectionAction.ASK_USER.value, ReflectionAction.ABSTAIN.value} or decision in {
                ExecutionDecision.ASK_FOR_CLARIFICATION,
                ExecutionDecision.ABSTAIN,
                ExecutionDecision.RETRY_RETRIEVAL,
            }
        return False

    def _phase_advice(self, context: ExecutionContext, phase: WorkflowPhase) -> ReflectionAdvice | None:
        if phase is WorkflowPhase.REFLECTION:
            advice = getattr(context.reflection_result, "advice", None)
        elif phase is WorkflowPhase.OUTPUT_VALIDATION:
            advice = getattr(context.output_validation, "advice", None)
        else:
            return None
        return advice if isinstance(advice, ReflectionAdvice) else None

    def _correction_phases(
        self,
        context: ExecutionContext,
        advice: ReflectionAdvice,
        source_phase: WorkflowPhase,
    ) -> list[WorkflowPhase]:
        stage = "post_execution" if source_phase is WorkflowPhase.OUTPUT_VALIDATION else "pre_execution"
        attempt = self.correction_loop.review(context, advice, stage)
        if not attempt.accepted:
            return []
        action = advice.action
        if action in {
            ReflectionAction.REVISE_QUERY,
            ReflectionAction.BROADEN_QUERY,
            ReflectionAction.PIVOT_ENTITY,
        }:
            context.metadata["retrieval_query_override"] = attempt.after_query
            return self._query_correction_route(context, post_execution=stage == "post_execution")
        if action is ReflectionAction.REPLAN:
            context.metadata["replan_reason"] = advice.reason_code or advice.correction_proposal
            phases = [
                WorkflowPhase.PLANNING,
                WorkflowPhase.WORLD_SIMULATION,
                WorkflowPhase.UNCERTAINTY_ESTIMATION,
                WorkflowPhase.EXECUTION_DECISION,
                WorkflowPhase.REFLECTION,
            ]
            if stage == "post_execution":
                phases.extend(self._execution_tail(context))
            return phases
        if action is ReflectionAction.REGENERATE:
            return [WorkflowPhase.ANSWER_GENERATION, WorkflowPhase.OUTPUT_VALIDATION]
        if action is ReflectionAction.RETRY_TRANSIENT_FAILURE:
            if stage == "post_execution" and context.action_result is not None:
                return [WorkflowPhase.ACTION, WorkflowPhase.OUTPUT_VALIDATION]
            context.metadata["retrieval_query_override"] = attempt.after_query
            return self._query_correction_route(context, post_execution=stage == "post_execution")
        if action in {ReflectionAction.ASK_USER, ReflectionAction.ABSTAIN}:
            context.metadata["correction_terminal_action"] = action.value
        return []

    def _query_correction_route(self, context: ExecutionContext, *, post_execution: bool) -> list[WorkflowPhase]:
        profile = str(context.metadata.get("profile", ""))
        if profile != "prima_full":
            return [
                WorkflowPhase.EVIDENCE_ACQUISITION,
                WorkflowPhase.ANSWER_GENERATION,
                WorkflowPhase.OUTPUT_VALIDATION,
            ]
        phases = [
            WorkflowPhase.EVIDENCE_ACQUISITION,
            WorkflowPhase.PLANNING,
            WorkflowPhase.WORLD_SIMULATION,
            WorkflowPhase.UNCERTAINTY_ESTIMATION,
            WorkflowPhase.EXECUTION_DECISION,
            WorkflowPhase.REFLECTION,
        ]
        if post_execution:
            phases.extend(self._execution_tail(context))
        return phases

    def _execution_tail(self, context: ExecutionContext) -> list[WorkflowPhase]:
        phases = [WorkflowPhase.ACTION]
        if str(context.metadata.get("task_kind")) != "tool_request":
            phases.append(WorkflowPhase.ANSWER_GENERATION)
        phases.append(WorkflowPhase.OUTPUT_VALIDATION)
        return phases

    async def _check_cancelled(self, context: ExecutionContext) -> None:
        if context.cancellation_requested:
            raise WorkflowCancelledError("Workflow cancellation requested.")

    async def _publish(
        self,
        context: ExecutionContext,
        event_type: WorkflowEventType,
        phase: WorkflowPhase | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        await self.event_bus.publish(
            WorkflowEvent(
                event_type=event_type,
                execution_id=context.execution_id,
                status=context.workflow_state.status,
                phase=phase,
                payload=payload or {},
            )
        )
