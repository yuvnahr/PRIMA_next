"""Bounded local worker for asynchronous memory maintenance."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from events.event import Event
from events.event_bus import EventBus
from events.event_types import EventType
from events.maintenance_events import MAINTENANCE_EVENT_TYPES, maintenance_event

MaintenanceHandler = Callable[[Event], None | Awaitable[None]]


class MaintenanceMode(str, Enum):
    """Benchmark-visible maintenance consistency modes."""

    DISABLED = "disabled"
    EVENTUAL = "eventual"
    FLUSH_AFTER_CONVERSATION = "flush_after_conversation"
    FLUSH_BEFORE_QUESTION = "flush_before_question"
    FLUSH_BEFORE_FINALIZATION = "flush_before_finalization"


class MaintenanceBarrier(str, Enum):
    """Lifecycle points at which a benchmark may request a deterministic flush."""

    AFTER_CONVERSATION = "after_conversation"
    BEFORE_QUESTION = "before_question"
    BEFORE_FINALIZATION = "before_finalization"


@dataclass(frozen=True, slots=True)
class MaintenanceFailure:
    """Persisted terminal failure for one maintenance event."""

    event_id: str
    event_type: str
    memory_id: str
    attempts: int
    error: str
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the failure into a versioned JSON-compatible record."""

        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "memory_id": self.memory_id,
            "attempts": self.attempts,
            "error": self.error,
        }


class MaintenanceFailureStore(Protocol):
    """Storage boundary for terminal maintenance failures."""

    def append(self, failure: MaintenanceFailure) -> None:
        """Persist one failure."""

    def all(self) -> tuple[MaintenanceFailure, ...]:
        """Return persisted failures in insertion order."""


class InMemoryMaintenanceFailureStore:
    """Deterministic failure store for tests and benchmark isolation."""

    def __init__(self) -> None:
        self._failures: list[MaintenanceFailure] = []

    def append(self, failure: MaintenanceFailure) -> None:
        """Retain one failure for the runtime lifetime."""

        self._failures.append(failure)

    def all(self) -> tuple[MaintenanceFailure, ...]:
        """Return retained failures in insertion order."""

        return tuple(self._failures)


class JsonlMaintenanceFailureStore(InMemoryMaintenanceFailureStore):
    """Append-only durable failure store used by production runtimes."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                payload = json.loads(line)
                self._failures.append(
                    MaintenanceFailure(
                        event_id=str(payload["event_id"]),
                        event_type=str(payload["event_type"]),
                        memory_id=str(payload["memory_id"]),
                        attempts=int(payload["attempts"]),
                        error=str(payload["error"]),
                        schema_version=str(payload["schema_version"]),
                    )
                )

    def append(self, failure: MaintenanceFailure) -> None:
        """Retain and append one versioned JSONL failure record."""

        super().append(failure)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(failure.to_dict(), sort_keys=True) + "\n")


class BackgroundMaintenanceSupervisor:
    """Run cold-path work outside request execution with bounded resources."""

    def __init__(
        self,
        handler: MaintenanceHandler,
        *,
        event_bus: EventBus | None = None,
        failure_store: MaintenanceFailureStore | None = None,
        max_queue_size: int = 128,
        max_retries: int = 2,
        enabled: bool = True,
    ) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive.")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        self.handler = handler
        self.event_bus = event_bus or EventBus()
        self.failure_store = failure_store or InMemoryMaintenanceFailureStore()
        self.max_retries = max_retries
        self.enabled = enabled
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        self._worker: asyncio.Task[None] | None = None
        self._completed_ids: set[str] = set()
        self._terminal_ids: set[str] = set()
        self._pending_ids: set[str] = set()
        self._retry_count = 0
        self._duplicate_count = 0
        self._rejected_count = 0

    @property
    def running(self) -> bool:
        """Return whether the worker task is active."""

        return self._worker is not None and not self._worker.done()

    def enqueue(self, event: Event) -> bool:
        """Queue work without awaiting; return false for duplicates or saturation."""

        if event.event_type not in MAINTENANCE_EVENT_TYPES:
            raise ValueError(f"Unsupported maintenance event: {event.event_type.value}.")
        if not self.enabled:
            return False
        if (
            event.event_id in self._completed_ids
            or event.event_id in self._terminal_ids
            or event.event_id in self._pending_ids
        ):
            self._duplicate_count += 1
            return False
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._rejected_count += 1
            self._record_failure(event, 0, "maintenance_queue_full")
            return False
        self._pending_ids.add(event.event_id)
        return True

    async def start(self) -> None:
        """Start one worker; repeated calls are idempotent."""

        if not self.enabled or self.running:
            return
        self._worker = asyncio.create_task(self._run(), name="prima-memory-maintenance")

    async def flush(self) -> None:
        """Wait until all work queued before the barrier is complete."""

        if not self.enabled:
            return
        await self.start()
        await self._queue.join()
        owner = getattr(self.handler, "__self__", None)
        flush = getattr(owner, "flush", None)
        if callable(flush):
            result = flush()
            if inspect.isawaitable(result):
                await result

    async def stop(self, *, graceful: bool = True) -> None:
        """Stop the worker, optionally flushing first; cancellation preserves current work."""

        if graceful and (self.running or not self._queue.empty()):
            await self.flush()
        worker = self._worker
        self._worker = None
        if worker is None or worker.done():
            self._rebind_queue()
            return
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        self._rebind_queue()

    async def apply_barrier(self, mode: MaintenanceMode, barrier: MaintenanceBarrier) -> bool:
        """Apply one benchmark consistency barrier and report whether it flushed."""

        should_flush = {
            MaintenanceMode.FLUSH_AFTER_CONVERSATION: MaintenanceBarrier.AFTER_CONVERSATION,
            MaintenanceMode.FLUSH_BEFORE_QUESTION: MaintenanceBarrier.BEFORE_QUESTION,
            MaintenanceMode.FLUSH_BEFORE_FINALIZATION: MaintenanceBarrier.BEFORE_FINALIZATION,
        }.get(mode) is barrier
        if not self.enabled or mode is MaintenanceMode.DISABLED:
            return False
        if mode is MaintenanceMode.EVENTUAL and barrier is MaintenanceBarrier.BEFORE_FINALIZATION:
            should_flush = True
        if should_flush:
            await self.flush()
        return should_flush

    def diagnostics(self) -> dict[str, Any]:
        """Return a versioned snapshot safe for runtime and benchmark manifests."""

        failures = self.failure_store.all()
        return {
            "schema_version": "1.0",
            "running": self.running,
            "enabled": self.enabled,
            "queue_size": self._queue.qsize(),
            "queue_capacity": self._queue.maxsize,
            "completed_count": len(self._completed_ids),
            "retry_count": self._retry_count,
            "duplicate_count": self._duplicate_count,
            "rejected_count": self._rejected_count,
            "failure_count": len(failures),
            "failures": [failure.to_dict() for failure in failures],
        }

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self._handle(event)
            except asyncio.CancelledError:
                try:
                    self._queue.put_nowait(event)
                except asyncio.QueueFull:
                    self._pending_ids.discard(event.event_id)
                    self._record_failure(event, 0, "maintenance_cancelled_queue_full")
                raise
            finally:
                self._queue.task_done()

    async def _handle(self, event: Event) -> None:
        attempts = 0
        while True:
            attempts += 1
            try:
                result = self.handler(event)
                if inspect.isawaitable(result):
                    await result
                self._completed_ids.add(event.event_id)
                self._pending_ids.discard(event.event_id)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempts <= self.max_retries:
                    self._retry_count += 1
                    continue
                self._pending_ids.discard(event.event_id)
                self._record_failure(event, attempts, f"{type(exc).__name__}: {exc}")
                return

    def _record_failure(self, event: Event, attempts: int, error: str) -> None:
        self._terminal_ids.add(event.event_id)
        memory_id = str(event.payload.get("memory_id", ""))
        failure = MaintenanceFailure(event.event_id, event.event_type.value, memory_id, attempts, error)
        self.failure_store.append(failure)
        failed_event = maintenance_event(
            EventType.MAINTENANCE_FAILED,
            memory_id=memory_id or "unknown",
            source="background_maintenance_supervisor",
            execution_id=event.execution_id,
            causation_id=event.event_id,
            payload={"attempts": attempts, "error": error},
        )
        self.event_bus.store.append(failed_event)

    def _rebind_queue(self) -> None:
        """Move pending work to a fresh queue for a later event-loop lifecycle."""

        pending: list[Event] = []
        while True:
            try:
                pending.append(self._queue.get_nowait())
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        self._queue = asyncio.Queue(maxsize=self._queue.maxsize)
        for event in pending:
            self._queue.put_nowait(event)
