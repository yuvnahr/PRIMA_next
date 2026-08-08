from __future__ import annotations

import asyncio

import pytest

from events.event import Event
from events.event_types import EventType
from events.maintenance_events import maintenance_event
from memory.maintenance.background_supervisor import (
    BackgroundMaintenanceSupervisor,
    JsonlMaintenanceFailureStore,
    MaintenanceBarrier,
    MaintenanceMode,
)
from memory.memory_repository import InMemoryMemoryRepository
from runtime.contracts import ExecutionProfile, PrimaRequest, RuntimeComponent, TaskKind
from runtime.prima_runtime import PrimaRuntime


def _event(event_id: str = "maintenance-1", memory_id: str = "memory-1") -> Event:
    return Event(
        event_type=EventType.MEMORY_ADMITTED,
        source="test",
        payload={"schema_version": "1.0", "memory_id": memory_id},
        event_id=event_id,
        topic="memory_maintenance",
    )


def test_queue_is_bounded_and_duplicate_events_are_idempotent() -> None:
    handled: list[str] = []
    supervisor = BackgroundMaintenanceSupervisor(
        lambda event: handled.append(event.event_id), max_queue_size=1
    )
    first = _event()

    assert supervisor.enqueue(first)
    assert not supervisor.enqueue(first)
    assert not supervisor.enqueue(_event("maintenance-2"))
    assert supervisor.diagnostics()["queue_size"] == 1
    assert supervisor.diagnostics()["duplicate_count"] == 1
    assert supervisor.diagnostics()["failure_count"] == 1

    async def drain() -> None:
        await supervisor.flush()
        await supervisor.stop()

    asyncio.run(drain())
    assert handled == [first.event_id]


def test_retry_terminal_failure_and_restart_are_traceable() -> None:
    attempts: dict[str, int] = {}

    def handler(event: Event) -> None:
        attempts[event.event_id] = attempts.get(event.event_id, 0) + 1
        if event.event_id == "retry" and attempts[event.event_id] == 1:
            raise RuntimeError("transient")
        if event.event_id == "failure":
            raise ValueError("permanent")

    supervisor = BackgroundMaintenanceSupervisor(handler, max_retries=1)
    assert supervisor.enqueue(_event("retry"))
    assert supervisor.enqueue(_event("failure"))

    async def run() -> None:
        await supervisor.flush()
        await supervisor.stop()
        await supervisor.start()
        await supervisor.stop()

    asyncio.run(run())
    diagnostics = supervisor.diagnostics()
    assert attempts == {"retry": 2, "failure": 2}
    assert diagnostics["completed_count"] == 1
    assert diagnostics["retry_count"] == 2
    assert diagnostics["failure_count"] == 1
    assert diagnostics["failures"][0]["event_id"] == "failure"
    assert supervisor.event_bus.store.filter(event_type=EventType.MAINTENANCE_FAILED)
    assert not supervisor.enqueue(_event("failure"))


def test_failure_store_survives_restart(tmp_path) -> None:
    path = tmp_path / "maintenance_failures.jsonl"
    store = JsonlMaintenanceFailureStore(path)

    async def run() -> None:
        supervisor = BackgroundMaintenanceSupervisor(
            lambda _event: (_ for _ in ()).throw(RuntimeError("broken")),
            failure_store=store,
            max_retries=0,
        )
        assert supervisor.enqueue(_event("persisted"))
        await supervisor.flush()
        await supervisor.stop()

    asyncio.run(run())
    restored = JsonlMaintenanceFailureStore(path)
    assert restored.all()[0].event_id == "persisted"
    assert restored.all()[0].schema_version == "1.0"


def test_worker_cancellation_preserves_in_flight_event_for_restart() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def handler(_event: Event) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()

    supervisor = BackgroundMaintenanceSupervisor(handler)
    assert supervisor.enqueue(_event())

    async def run() -> None:
        await supervisor.start()
        await started.wait()
        await supervisor.stop(graceful=False)
        assert supervisor.diagnostics()["queue_size"] == 1
        release.set()
        await supervisor.flush()
        await supervisor.stop()

    asyncio.run(run())
    assert calls == 2
    assert supervisor.diagnostics()["completed_count"] == 1


def test_barrier_modes_flush_only_at_their_declared_boundary() -> None:
    handled: list[str] = []
    supervisor = BackgroundMaintenanceSupervisor(lambda event: handled.append(event.event_id))
    assert supervisor.enqueue(_event())

    async def run() -> None:
        assert not await supervisor.apply_barrier(
            MaintenanceMode.FLUSH_BEFORE_QUESTION, MaintenanceBarrier.AFTER_CONVERSATION
        )
        assert handled == []
        assert await supervisor.apply_barrier(
            MaintenanceMode.FLUSH_BEFORE_QUESTION, MaintenanceBarrier.BEFORE_QUESTION
        )
        await supervisor.stop()

    asyncio.run(run())
    assert handled == ["maintenance-1"]


def test_sync_style_barriers_can_restart_on_new_event_loops() -> None:
    handled: list[str] = []
    supervisor = BackgroundMaintenanceSupervisor(lambda event: handled.append(event.event_id))

    for event_id in ("loop-1", "loop-2"):
        assert supervisor.enqueue(_event(event_id))

        async def drain() -> None:
            await supervisor.flush()
            await supervisor.stop()

        asyncio.run(drain())

    assert handled == ["loop-1", "loop-2"]


def test_runtime_enqueues_and_flushes_the_real_cold_path(tmp_path) -> None:
    runtime = PrimaRuntime(
        memory_repository=InMemoryMemoryRepository(),
        log_path=tmp_path / "runtime.log",
    )

    async def run() -> None:
        await runtime.start_maintenance()
        response = await runtime.execute(
            PrimaRequest(
                task_kind=TaskKind.DOCUMENT_INGESTION,
                profile=ExecutionProfile.INGESTION_ONLY,
                input_text="A durable semantic memory about Ada Lovelace.",
            )
        )
        assert RuntimeComponent.MAINTENANCE_EVENTS in response.diagnostics.executed_components
        await runtime.flush_maintenance()
        await runtime.stop_maintenance()

    asyncio.run(run())
    event_types = {event.event_type for event in runtime.maintenance_supervisor.event_bus.events}
    assert {
        EventType.MEMORY_ADMITTED,
        EventType.MEMORY_ENCODED,
        EventType.SALIENCE_SCORED,
        EventType.SHORT_TERM_BUFFERED,
        EventType.CONSOLIDATION_REQUESTED,
        EventType.CONSOLIDATION_COMPLETED,
        EventType.ABSTRACTION_REQUESTED,
        EventType.ABSTRACTION_COMPLETED,
        EventType.GRAPH_INDEX_UPDATED,
        EventType.DECAY_APPLIED,
    } <= event_types
    assert runtime.maintenance_supervisor.diagnostics()["failure_count"] == 0


def test_maintenance_failures_are_visible_in_later_runtime_diagnostics(tmp_path) -> None:
    def fail(_event: Event) -> None:
        raise RuntimeError("cold-path failure")

    supervisor = BackgroundMaintenanceSupervisor(fail, max_retries=0)
    runtime = PrimaRuntime(
        memory_repository=InMemoryMemoryRepository(),
        maintenance_supervisor=supervisor,
        log_path=tmp_path / "runtime.log",
    )

    async def run() -> None:
        await runtime.start_maintenance()
        await runtime.execute(
            PrimaRequest(
                task_kind=TaskKind.DOCUMENT_INGESTION,
                profile=ExecutionProfile.INGESTION_ONLY,
                input_text="A memory that triggers maintenance.",
            )
        )
        await runtime.flush_maintenance()
        response = await runtime.execute(
            PrimaRequest(
                task_kind=TaskKind.EMOTION_CLASSIFICATION,
                profile=ExecutionProfile.AFFECT_ONLY,
                input_text="I feel calm.",
            )
        )
        assert response.diagnostics.maintenance["failure_count"] == 1
        assert response.diagnostics.maintenance["failures"][0]["error"].endswith("cold-path failure")
        await runtime.stop_maintenance()

    asyncio.run(run())


def test_hot_path_does_not_await_cold_path_handler(tmp_path) -> None:
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def block(_event: Event) -> None:
        handler_started.set()
        await release_handler.wait()

    supervisor = BackgroundMaintenanceSupervisor(block)
    runtime = PrimaRuntime(
        memory_repository=InMemoryMemoryRepository(),
        maintenance_supervisor=supervisor,
        log_path=tmp_path / "runtime.log",
    )

    async def run() -> None:
        await runtime.start_maintenance()
        response = await asyncio.wait_for(
            runtime.execute(
                PrimaRequest(
                    task_kind=TaskKind.DOCUMENT_INGESTION,
                    profile=ExecutionProfile.INGESTION_ONLY,
                    input_text="The request must finish while cold work is blocked.",
                )
            ),
            timeout=2,
        )
        await handler_started.wait()
        assert response.output_data["memory_id"]
        assert supervisor.diagnostics()["completed_count"] == 0
        release_handler.set()
        await runtime.flush_maintenance()
        await runtime.stop_maintenance()

    asyncio.run(run())


def test_disabled_maintenance_is_explicitly_skipped(tmp_path) -> None:
    runtime = PrimaRuntime(
        memory_repository=InMemoryMemoryRepository(),
        maintenance_enabled=False,
        log_path=tmp_path / "runtime.log",
    )
    response = asyncio.run(
        runtime.execute(
            PrimaRequest(
                task_kind=TaskKind.DOCUMENT_INGESTION,
                profile=ExecutionProfile.INGESTION_ONLY,
                input_text="Maintenance is disabled for this run.",
            )
        )
    )

    assert RuntimeComponent.MAINTENANCE_EVENTS in response.diagnostics.skipped_components
    details = response.diagnostics.component_details[RuntimeComponent.MAINTENANCE_EVENTS.value]
    assert details["status"] == "disabled"
    assert details["reason"] == "maintenance_disabled"


def test_typed_maintenance_events_reject_invalid_kinds() -> None:
    with pytest.raises(ValueError, match="not a maintenance event"):
        maintenance_event(
            EventType.CUSTOM,
            memory_id="memory-1",
            source="test",
        )
