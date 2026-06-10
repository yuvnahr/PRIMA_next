import unittest

from events import Event, EventBus, EventPublisher, EventType
from workflow.workflow_events import WorkflowEvent, WorkflowEventBus, WorkflowEventType
from workflow.workflow_state import WorkflowPhase, WorkflowStatus


class EventsPhase6Test(unittest.IsolatedAsyncioTestCase):
    async def test_event_bus_publishes_to_matching_async_subscribers_and_stores_events(self) -> None:
        bus = EventBus()
        received: list[str] = []

        async def on_memory_created(event: Event) -> None:
            received.append(str(event.payload["memory_id"]))

        bus.subscribe(
            on_memory_created,
            event_types=(EventType.MEMORY_CREATED,),
            topics=("memory",),
        )

        result = await bus.publish(
            Event.create(
                EventType.MEMORY_CREATED,
                source="memory.repository",
                topic="memory",
                payload={"memory_id": "mem_1"},
            )
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.delivered_count, 1)
        self.assertEqual(received, ["mem_1"])
        self.assertEqual(bus.store.all()[0].event_type, EventType.MEMORY_CREATED)

    async def test_event_bus_filters_by_type_topic_and_source(self) -> None:
        bus = EventBus()
        received: list[str] = []

        def on_reflection(event: Event) -> None:
            received.append(event.source)

        bus.subscribe(
            on_reflection,
            event_types=(EventType.REFLECTION_TRIGGERED,),
            topics=("reflection",),
            sources=("reflection.engine",),
        )

        await bus.publish(
            Event.create(
                EventType.REFLECTION_TRIGGERED,
                source="memory.repository",
                topic="reflection",
                payload={"reason": "ignored source"},
            )
        )
        await bus.publish(
            Event.create(
                EventType.REFLECTION_TRIGGERED,
                source="reflection.engine",
                topic="reflection",
                payload={"reason": "low confidence"},
            )
        )

        self.assertEqual(received, ["reflection.engine"])

    async def test_publisher_helpers_create_canonical_events(self) -> None:
        bus = EventBus()
        publisher = EventPublisher(source="tool.executor", topic="tools")

        result = await bus.publish(publisher.tool_executed("safe_lookup", "success"))

        self.assertEqual(result.event.event_type, EventType.TOOL_EXECUTED)
        self.assertEqual(result.event.payload["tool_name"], "safe_lookup")
        self.assertEqual(bus.store.filter(event_type=EventType.TOOL_EXECUTED)[0].topic, "tools")

    async def test_workflow_event_bus_remains_backward_compatible(self) -> None:
        bus = WorkflowEventBus()
        received: list[WorkflowEventType] = []

        def on_workflow(event: WorkflowEvent) -> None:
            received.append(event.event_type)

        bus.subscribe(on_workflow)
        await bus.publish(
            WorkflowEvent(
                event_type=WorkflowEventType.PHASE_COMPLETED,
                execution_id="exec_1",
                status=WorkflowStatus.RUNNING,
                phase=WorkflowPhase.PLANNING,
                payload={"ok": True},
            )
        )

        self.assertEqual(received, [WorkflowEventType.PHASE_COMPLETED])
        self.assertEqual(bus.events[0].event_type, WorkflowEventType.PHASE_COMPLETED)
        self.assertEqual(bus.events[0].phase, WorkflowPhase.PLANNING)
        self.assertEqual(bus.event_bus.store.all()[0].event_type, EventType.WORKFLOW_PHASE_COMPLETED)


if __name__ == "__main__":
    unittest.main()
