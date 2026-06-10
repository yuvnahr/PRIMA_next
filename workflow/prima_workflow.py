"""High-level PRIMA workflow facade and default controller adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from affect.affect_engine import DynamicAffectEngine
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_controller import RetrievalController
from memory.retrieval.retrieval_request import RetrievalRequest
from planning import PlanningContext, TaskPlanner
from planning.plan import Plan
from reflection.reflection_context import ReflectionContext
from reflection.reflection_engine import ReflectionEngine
from workflow.controller_registry import ControllerRegistry
from workflow.execution_context import ExecutionContext
from workflow.orchestration_engine import OrchestrationEngine, RetryPolicy
from workflow.task_router import TaskRouter
from workflow.workflow_events import WorkflowEventBus
from workflow.workflow_state import WorkflowPhase


@dataclass(slots=True)
class AffectController:
    """Workflow controller for affect analysis."""

    affect_engine: DynamicAffectEngine
    phase: WorkflowPhase = WorkflowPhase.AFFECT

    async def execute(self, context: ExecutionContext) -> Any:
        """Run affect analysis through the workflow."""
        return self.affect_engine.process(context.user_input, cognitive_state=context.cognitive_state)


@dataclass(slots=True)
class MemoryRetrievalController:
    """Workflow controller for memory retrieval."""

    retrieval_controller: RetrievalController
    top_k: int = 5
    phase: WorkflowPhase = WorkflowPhase.MEMORY_RETRIEVAL

    async def execute(self, context: ExecutionContext) -> Any:
        """Run memory retrieval using workflow-owned context."""
        affect_priors = {}
        affect_signals = ()
        if context.affect_update is not None:
            affect_priors = dict(getattr(context.affect_update, "retrieval_priors", {}))
            affect_signals = tuple(getattr(context.affect_update, "reflection_signals", ()))
        request = RetrievalRequest(
            query=context.user_input,
            memory_types=tuple(MemoryType),
            top_k=self.top_k,
            state_filter={
                "goal_state": context.cognitive_state.goal_state,
                "task_state": context.cognitive_state.task_state,
                "confidence_state": context.cognitive_state.confidence_state,
                "environment_state": context.cognitive_state.environment_state,
            },
            affective_context={"retrieval_priors": affect_priors, "reflection_signal_count": len(affect_signals)},
        )
        return self.retrieval_controller.retrieve(request)


@dataclass(slots=True)
class PlanningController:
    """Workflow controller for planning."""

    planner: TaskPlanner = field(default_factory=TaskPlanner)
    phase: WorkflowPhase = WorkflowPhase.PLANNING

    async def execute(self, context: ExecutionContext) -> Plan:
        """Create or revise a pure reasoning plan from workflow-routed context."""
        planning_context = PlanningContext.from_subsystem_outputs(
            objective=context.user_input,
            cognitive_state=context.cognitive_state,
            retrieval_response=context.retrieval_response,
            affect_update=context.affect_update,
            reflection_signals=tuple(context.metadata.get("reflection_signals", ())),
            metadata={"execution_id": context.execution_id},
        )
        replan_reason = context.metadata.get("replan_reason")
        if isinstance(context.plan, Plan) and replan_reason:
            return self.planner.replan(planning_context, context.plan, str(replan_reason))
        return self.planner.create_plan(planning_context)


@dataclass(slots=True)
class ReflectionController:
    """Workflow controller for adaptive reflection."""

    reflection_engine: ReflectionEngine
    phase: WorkflowPhase = WorkflowPhase.REFLECTION

    async def execute(self, context: ExecutionContext) -> Any:
        """Run reflection using only workflow-routed subsystem outputs."""
        retrieval_confidence = getattr(context.retrieval_response, "confidence", None)
        retrieved_memories = tuple(getattr(context.retrieval_response, "results", ()))
        affect_signals = tuple(getattr(context.affect_update, "reflection_signals", ())) if context.affect_update else ()
        failure_metadata = {
            "reason": "workflow reflection checkpoint",
            "severity": 0.4,
        }
        if retrieval_confidence and retrieval_confidence.confidence < 0.4:
            failure_metadata = {
                "reason": "low confidence retrieval with ambiguous memories",
                "severity": 0.8,
            }
        reflection_history = tuple(context.metadata.get("reflection_history", ()))
        reflection_context = ReflectionContext(
            query=context.user_input,
            retrieved_memories=retrieved_memories,
            retrieval_confidence=retrieval_confidence,
            cognitive_state=context.cognitive_state,
            emotional_state=context.cognitive_state.emotional_state,
            reflection_history=reflection_history,
            failure_metadata=failure_metadata,
            affect_signals=affect_signals,
        )
        return self.reflection_engine.evaluate(reflection_context)


@dataclass(slots=True)
class ActionController:
    """Workflow controller for action preparation."""

    phase: WorkflowPhase = WorkflowPhase.ACTION

    async def execute(self, context: ExecutionContext) -> dict[str, Any]:
        """Produce a structured action request without executing external effects."""
        execution_intent = getattr(context.plan, "execution_intent", None)
        intent_type = getattr(getattr(execution_intent, "intent_type", None), "value", "respond")
        plan_payload = context.plan.to_dict() if hasattr(context.plan, "to_dict") else context.plan
        return {
            "action_type": intent_type,
            "requires_external_tool": bool(getattr(execution_intent, "requires_external_tool", False)),
            "plan": plan_payload,
            "reflection_triggered": bool(getattr(context.reflection_result, "should_reflect", False)),
        }


@dataclass(slots=True)
class OutputController:
    """Workflow controller for final output shaping."""

    phase: WorkflowPhase = WorkflowPhase.OUTPUT

    async def execute(self, context: ExecutionContext) -> dict[str, Any]:
        """Build final workflow output."""
        return {
            "text": context.user_input,
            "execution_id": context.execution_id,
            "dominant_emotion": getattr(context.affect_update.profile, "dominant_emotion", "neutral")
            if context.affect_update
            else "neutral",
            "memory_count": len(getattr(context.retrieval_response, "results", ())) if context.retrieval_response else 0,
            "plan_status": getattr(getattr(context.plan, "status", None), "value", None),
            "reflection_triggered": bool(getattr(context.reflection_result, "should_reflect", False)),
        }


class PrimaWorkflow:
    """Facade for executing the PRIMA cognitive control bus."""

    def __init__(
        self,
        registry: ControllerRegistry,
        router: TaskRouter | None = None,
        event_bus: WorkflowEventBus | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.engine = OrchestrationEngine(
            registry=registry,
            router=router,
            event_bus=event_bus,
            retry_policy=retry_policy,
        )

    async def run(self, user_input: str, context: ExecutionContext | None = None) -> ExecutionContext:
        """Run one cognitive workflow execution."""
        execution_context = context or ExecutionContext(user_input=user_input)
        if context is not None:
            execution_context.user_input = user_input
        return await self.engine.execute(execution_context)

    async def cancel(self, context: ExecutionContext) -> None:
        """Request cancellation for a running context."""
        await self.engine.cancel(context)

    @classmethod
    def from_controllers(
        cls,
        affect_engine: DynamicAffectEngine,
        retrieval_controller: RetrievalController,
        reflection_engine: ReflectionEngine,
        event_bus: WorkflowEventBus | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> "PrimaWorkflow":
        """Create a workflow with default controller adapters."""
        registry = ControllerRegistry(
            controllers={
                WorkflowPhase.AFFECT: AffectController(affect_engine),
                WorkflowPhase.MEMORY_RETRIEVAL: MemoryRetrievalController(retrieval_controller),
                WorkflowPhase.PLANNING: PlanningController(),
                WorkflowPhase.REFLECTION: ReflectionController(reflection_engine),
                WorkflowPhase.ACTION: ActionController(),
                WorkflowPhase.OUTPUT: OutputController(),
            }
        )
        return cls(registry=registry, event_bus=event_bus, retry_policy=retry_policy)
