"""Cognitive Control Bus orchestration engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from workflow.controller_registry import ControllerRegistry
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


class OrchestrationEngine:
    """Owns the workflow execution lifecycle."""

    def __init__(
        self,
        registry: ControllerRegistry,
        router: TaskRouter | None = None,
        event_bus: WorkflowEventBus | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.router = router or TaskRouter()
        self.event_bus = event_bus or WorkflowEventBus()
        self.retry_policy = retry_policy or RetryPolicy()

    async def execute(self, context: ExecutionContext) -> ExecutionContext:
        """Execute the routed cognitive workflow."""
        context.workflow_state.status = WorkflowStatus.RUNNING
        await self._publish(context, WorkflowEventType.WORKFLOW_STARTED)
        try:
            for phase in self.router.route(context).phases:
                await self._check_cancelled(context)
                context.workflow_state.current_phase = phase
                await self._publish(context, WorkflowEventType.PHASE_STARTED, phase)
                await self._execute_phase_with_retries(context, phase)
                context.workflow_state.mark_phase_complete(phase)
                await self._publish(context, WorkflowEventType.PHASE_COMPLETED, phase)

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
            raise
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
        if phase == WorkflowPhase.AFFECT:
            context.affect_update = result
        elif phase == WorkflowPhase.MEMORY_RETRIEVAL:
            context.retrieval_response = result
        elif phase == WorkflowPhase.PLANNING:
            context.plan = result
        elif phase == WorkflowPhase.REFLECTION:
            context.reflection_result = result
        elif phase == WorkflowPhase.ACTION:
            context.action_result = result
        elif phase == WorkflowPhase.OUTPUT:
            context.output = result
        context.workflow_state.outputs[phase.value] = result

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
