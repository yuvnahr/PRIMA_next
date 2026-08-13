import unittest

from affect.affect_engine import DynamicAffectEngine
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_controller import RetrievalController
from planning import Plan
from reflection.reflection_engine import ReflectionEngine
from state.state_manager import InMemoryStateManager
from workflow.controller_registry import ControllerRegistry
from workflow.execution_context import ExecutionContext
from workflow.orchestration_engine import OrchestrationEngine, RetryPolicy
from workflow.prima_workflow import PrimaWorkflow
from workflow.workflow_events import WorkflowEventBus, WorkflowEventType
from workflow.workflow_state import WorkflowPhase, WorkflowStatus


class StaticController:
    def __init__(self, phase: WorkflowPhase, value: object) -> None:
        self.phase = phase
        self.value = value

    async def execute(self, context: ExecutionContext) -> object:
        return self.value


class FailingOnceController:
    phase = WorkflowPhase.PLANNING

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, context: ExecutionContext) -> object:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary planning failure")
        return {"planned": True}


class CancellingController:
    phase = WorkflowPhase.AFFECT

    async def execute(self, context: ExecutionContext) -> object:
        context.request_cancel()
        return {"cancel": "requested"}


class WorkflowPhase1Test(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_owns_lifecycle_routes_outputs_and_events(self) -> None:
        event_bus = WorkflowEventBus()
        registry = ControllerRegistry(
            controllers={
                WorkflowPhase.AFFECT: StaticController(WorkflowPhase.AFFECT, {"emotion": "joy"}),
                WorkflowPhase.MEMORY_RETRIEVAL: StaticController(WorkflowPhase.MEMORY_RETRIEVAL, {"memories": []}),
                WorkflowPhase.PLANNING: StaticController(WorkflowPhase.PLANNING, {"steps": ["act"]}),
                WorkflowPhase.REFLECTION: StaticController(WorkflowPhase.REFLECTION, {"reflect": False}),
                WorkflowPhase.ACTION: StaticController(WorkflowPhase.ACTION, {"action": "respond"}),
                WorkflowPhase.OUTPUT: StaticController(WorkflowPhase.OUTPUT, {"text": "done"}),
            }
        )
        context = await OrchestrationEngine(registry, event_bus=event_bus).execute(
            ExecutionContext(
                user_input="hello",
                metadata={"route": tuple(phase.value for phase in registry.controllers)},
            )
        )

        self.assertEqual(context.workflow_state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(context.affect_update, {"emotion": "joy"})
        self.assertEqual(context.output, {"text": "done"})
        self.assertEqual(
            context.workflow_state.completed_phases,
            [
                WorkflowPhase.AFFECT,
                WorkflowPhase.MEMORY_RETRIEVAL,
                WorkflowPhase.PLANNING,
                WorkflowPhase.REFLECTION,
                WorkflowPhase.ACTION,
                WorkflowPhase.OUTPUT,
            ],
        )
        self.assertEqual(event_bus.events[0].event_type, WorkflowEventType.WORKFLOW_STARTED)
        self.assertEqual(event_bus.events[-1].event_type, WorkflowEventType.WORKFLOW_COMPLETED)

    async def test_workflow_supports_retries(self) -> None:
        planner = FailingOnceController()
        registry = ControllerRegistry(
            controllers={
                WorkflowPhase.PLANNING: planner,
            }
        )
        context = ExecutionContext(user_input="plan only", metadata={"route": (WorkflowPhase.PLANNING.value,)})

        result = await OrchestrationEngine(
            registry,
            event_bus=WorkflowEventBus(),
            retry_policy=RetryPolicy(max_retries=1),
        ).execute(context)

        self.assertEqual(result.workflow_state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(planner.calls, 2)
        self.assertEqual(result.workflow_state.retry_counts[WorkflowPhase.PLANNING], 1)

    async def test_workflow_supports_cooperative_cancellation(self) -> None:
        registry = ControllerRegistry(
            controllers={
                WorkflowPhase.AFFECT: CancellingController(),
                WorkflowPhase.MEMORY_RETRIEVAL: StaticController(WorkflowPhase.MEMORY_RETRIEVAL, {"should": "not run"}),
            }
        )
        context = ExecutionContext(
            user_input="cancel",
            metadata={"route": (WorkflowPhase.AFFECT.value, WorkflowPhase.MEMORY_RETRIEVAL.value)},
        )

        result = await OrchestrationEngine(registry, event_bus=WorkflowEventBus()).execute(context)

        self.assertEqual(result.workflow_state.status, WorkflowStatus.CANCELLED)
        self.assertEqual(result.affect_update, {"cancel": "requested"})
        self.assertIsNone(result.retrieval_response)

    async def test_prima_workflow_routes_real_subsystem_outputs_through_context(self) -> None:
        repository = InMemoryMemoryRepository()
        repository.add(MemoryNote.create("I baked sourdough bread", memory_type=MemoryType.EPISODIC))
        workflow = PrimaWorkflow.from_controllers(
            affect_engine=DynamicAffectEngine(),
            retrieval_controller=RetrievalController(repository),
            reflection_engine=ReflectionEngine(),
            event_bus=WorkflowEventBus(),
            state_manager=InMemoryStateManager(),
        )

        context = await workflow.run(
            "I am nervous about sourdough bread tomorrow",
            ExecutionContext(
                user_input="I am nervous about sourdough bread tomorrow",
                metadata={
                    "route": tuple(phase.value for phase in (
                        WorkflowPhase.AFFECT,
                        WorkflowPhase.MEMORY_RETRIEVAL,
                        WorkflowPhase.PLANNING,
                        WorkflowPhase.REFLECTION,
                        WorkflowPhase.ACTION,
                        WorkflowPhase.OUTPUT,
                    )),
                },
            ),
        )

        self.assertEqual(context.workflow_state.status, WorkflowStatus.COMPLETED)
        self.assertIsNotNone(context.affect_update)
        self.assertIsNotNone(context.retrieval_response)
        self.assertIsInstance(context.plan, Plan)
        from typing import cast

        self.assertIsNotNone(context.reflection_result)
        self.assertIsNotNone(context.action_result)
        self.assertIsNotNone(context.output)
        action_result = cast(dict[str, object], context.action_result)
        plan_obj = cast(Plan, context.plan)
        self.assertFalse(cast(bool, action_result["requires_external_tool"]))
        plan_dict = cast(dict[str, object], action_result["plan"])
        self.assertEqual(plan_dict["plan_id"], plan_obj.plan_id)


if __name__ == "__main__":
    unittest.main()
