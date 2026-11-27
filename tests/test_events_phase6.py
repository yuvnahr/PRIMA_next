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


def test_overlapping_publications_dispatch_only_to_publisher_registrations():
    import asyncio

    async def run():
        shared = EventBus()
        a, b = WorkflowEventBus(shared), WorkflowEventBus(shared)
        entered, release = asyncio.Event(), asyncio.Event()
        seen = []

        async def barrier(event):
            if event.execution_id == "a":
                entered.set()
                await release.wait()

        shared.subscribe(barrier)
        a.subscribe(lambda event: seen.append(("a", event.execution_id)))
        b.subscribe(lambda event: seen.append(("b", event.execution_id)))

        def event(key):
            return WorkflowEvent(WorkflowEventType.WORKFLOW_STARTED, key, WorkflowStatus.RUNNING)

        first = asyncio.create_task(a.publish(event("a")))
        await entered.wait()
        second = await b.publish(event("b"))
        release.set()
        await first
        assert seen == [("b", "b"), ("a", "a")]
        assert len(shared.events) == 2 and second.delivered_count == 2

    asyncio.run(run())


def test_nested_publish_and_independent_local_failure_policy():
    import asyncio

    import pytest

    async def run(raise_errors):
        bus = WorkflowEventBus(EventBus(raise_subscriber_errors=raise_errors))
        seen = []

        async def first(event):
            seen.append(("first", event.execution_id))
            if event.execution_id == "outer":
                await bus.publish(WorkflowEvent(event.event_type, "inner", event.status))
                raise ValueError("local failure")

        bus.subscribe(first)
        bus.subscribe(lambda event: seen.append(("second", event.execution_id)))
        outer = WorkflowEvent(WorkflowEventType.WORKFLOW_STARTED, "outer", WorkflowStatus.RUNNING)
        if raise_errors:
            with pytest.raises(ValueError, match="local failure"):
                await bus.publish(outer)
        else:
            result = await bus.publish(outer)
            assert result.errors == ("local failure",) and result.delivered_count == 1
        assert seen[:3] == [("first", "outer"), ("first", "inner"), ("second", "inner")]
        assert (("second", "outer") in seen) != raise_errors
        assert len(bus.events) == 2

    asyncio.run(run(False))
    asyncio.run(run(True))
