"""Bounded local worker for asynchronous memory maintenance."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from events.event import Event
from events.event_bus import EventBus
from events.event_types import EventType
from events.maintenance_events import MAINTENANCE_EVENT_TYPES, maintenance_event

MaintenanceHandler = Callable[[Event], None | Awaitable[None]]


class MaintenanceMode(StrEnum):
    """Benchmark-visible maintenance consistency modes."""

    DISABLED = "disabled"
    EVENTUAL = "eventual"
    FLUSH_AFTER_CONVERSATION = "flush_after_conversation"
    FLUSH_BEFORE_QUESTION = "flush_before_question"
    FLUSH_BEFORE_FINALIZATION = "flush_before_finalization"


class MaintenanceBarrier(StrEnum):
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
        self._reload()

    def _reload(self) -> None:
        """A nonterminated, invalid last record is an interrupted append."""
        if not self.path.exists():
            return
        records = []
        with self.path.open("rb+") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                try:
                    payload = json.loads(line)
                except ValueError, UnicodeDecodeError:
                    if line.endswith(b"\n"):
                        raise
                    stream.seek(offset)
                    stream.truncate()
                    break
                records.append(MaintenanceFailure(**payload))
                if not line.endswith(b"\n"):
                    stream.write(b"\n")
        self._failures = records

    def append(self, failure: MaintenanceFailure) -> None:
        """Recover an ambiguous previous write by its stable event ID."""
        self._reload()
        if any(row.event_id == failure.event_id for row in self._failures):
            return
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(failure.to_dict(), sort_keys=True) + "\n")
        super().append(failure)


class MaintenanceUnavailable(RuntimeError):
    """Maintenance could not confirm completion inside the lifecycle bound."""


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
        timeout: float = 60.0,
    ) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive.")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Maintenance timeout must be finite and positive.")
        self.timeout = timeout
        self._unrecorded: dict[str, tuple[Event, MaintenanceFailure, Event]] = {}
        self._owner_flush: asyncio.Future[Any] | None = None
        self.handler = handler
        self.event_bus = event_bus or EventBus()
        self.failure_store = failure_store or InMemoryMaintenanceFailureStore()
        self.max_retries = max_retries
        self.enabled = enabled
        self._capacity = max_queue_size
        # Reserve one slot to return an interrupted in-flight event even when admission is full.
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size + 1)
        self._worker: asyncio.Task[None] | None = None
        self._completed_ids: set[str] = set()
        self._terminal_ids: set[str] = set()
        self._pending_ids: set[str] = set()
        self._stopping = False
        self._retry_count = 0
        self._duplicate_count = 0
        self._rejected_count = 0

    @property
    def running(self) -> bool:
        """Return whether the worker task is active."""

        return self._worker is not None and not self._worker.done()

    def enqueue(self, event: Event) -> bool:
        """Queue work without awaiting; return false for duplicates or saturation."""

        self._check_worker()
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
        if len(self._completed_ids) + len(self._terminal_ids) + len(self._pending_ids) >= 100_000:
            raise MaintenanceUnavailable("Run history capacity reached; export, stop and release history.")
        if self._queue.qsize() >= self._capacity:
            self._rejected_count += 1
            self._record_failure(event, 0, "maintenance_queue_full")
            return False
        self._queue.put_nowait(event)
        self._pending_ids.add(event.event_id)
        return True

    async def start(self) -> None:
        """Start one worker; repeated calls are idempotent."""

        self._check_worker()
        if not self.enabled or self.running:
            return
        self._stopping = False
        self._worker = asyncio.create_task(self._run(), name="prima-memory-maintenance")

    async def enqueue_wait(self, event: Event) -> bool:
        """Apply bounded backpressure instead of dropping admitted memory work."""
        if self.enabled and self._queue.qsize() >= self._capacity:
            await self.flush()
        return self.enqueue(event)

    async def flush(self) -> dict[str, Any]:
        """Finish queued work and return counts, including acknowledged terminal failures."""
        if not self.enabled:
            return self.diagnostics()
        await self.start()
        worker = self._worker
        if worker is None:
            raise MaintenanceUnavailable("Maintenance worker did not start.")
        deadline = asyncio.get_running_loop().time() + self.timeout
        join = asyncio.create_task(self._queue.join())
        try:
            done, _ = await asyncio.wait((join, worker), timeout=self.timeout, return_when=asyncio.FIRST_COMPLETED)
            self._check_worker()
            if join not in done:
                raise MaintenanceUnavailable("Maintenance queue barrier timed out.")
        finally:
            join.cancel()
            await asyncio.gather(join, return_exceptions=True)
        owner = getattr(self.handler, "__self__", None)
        flush = getattr(owner, "flush", None)
        if callable(flush):
            if self._owner_flush is None:
                result = flush()
                if not inspect.isawaitable(result):
                    return self.diagnostics()
                self._owner_flush = asyncio.ensure_future(result)
            task = self._owner_flush
            finished, _ = await asyncio.wait((task,), timeout=max(0.0, deadline - asyncio.get_running_loop().time()))
            if not finished:
                raise MaintenanceUnavailable("Maintenance batch flush timed out.")
            self._owner_flush = None
            task.result()
        self._check_worker()
        return self.diagnostics()

    def _check_worker(self) -> None:
        worker = self._worker
        if worker is not None and worker.done():
            error = None if worker.cancelled() else worker.exception()
            raise MaintenanceUnavailable(f"Maintenance worker stopped: {error or 'cancelled'}") from error
        if self._unrecorded:
            raise MaintenanceUnavailable("Maintenance failure publication needs recovery.")

    async def recover(self) -> None:
        """Repair the failed publication and resume queued work without repeating its handler."""
        for event_id in tuple(self._unrecorded):
            self._publish_failure(event_id)
        if self._worker is not None and self._worker.done():
            if not self._worker.cancelled():
                self._worker.exception()
            self._worker = None
            self._rebind_queue()
        await self.start()

    async def stop(self, *, graceful: bool = True) -> None:
        """Stop the worker, optionally flushing first; cancellation preserves current work."""

        deadline = asyncio.get_running_loop().time() + self.timeout
        try:
            self._check_worker()
            if graceful and (self.running or not self._queue.empty() or self._owner_flush is not None):
                await self.flush()
        finally:
            worker = self._worker
            if worker is not None and not worker.done():
                self._stopping = True
                worker.cancel()
                remaining = max(0.0, deadline - asyncio.get_running_loop().time())
                done, _ = await asyncio.wait((worker,), timeout=remaining)
                if not done:
                    raise MaintenanceUnavailable("Maintenance worker did not stop before deadline.")
            if worker is not None and not worker.cancelled() and worker.done():
                error = worker.exception()
                if error:
                    raise MaintenanceUnavailable(f"Maintenance worker stopped: {error}") from error
            self._worker = None
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
            if self._terminal_ids:
                raise MaintenanceUnavailable("Maintenance barrier contains terminal failures; inspect diagnostics.")
        return should_flush

    def release_history(self) -> None:
        """End the run's deduplication window after exporting diagnostics and stopping."""
        if self._worker is not None or self._pending_ids or self._unrecorded or self._owner_flush is not None:
            raise MaintenanceUnavailable("Stop and resolve outstanding work before releasing history.")
        self._completed_ids.clear()
        self._terminal_ids.clear()
        self._duplicate_count = self._rejected_count = self._retry_count = 0
        self.event_bus.store.clear()

    def diagnostics(self) -> dict[str, Any]:
        """Return a versioned snapshot safe for runtime and benchmark manifests."""

        failures = self.failure_store.all()
        return {
            "schema_version": "1.0",
            "running": self.running,
            "enabled": self.enabled,
            "queue_size": self._queue.qsize(),
            "queue_capacity": self._capacity,
            "unrecorded_event_ids": list(self._unrecorded),
            "completed_count": len(self._completed_ids),
            "terminal_count": len(self._terminal_ids),
            "pending_count": len(self._pending_ids),
            "retry_count": self._retry_count,
            "duplicate_count": self._duplicate_count,
            "rejected_count": self._rejected_count,
            "failure_count": len(failures),
            "failures": [failure.to_dict() for failure in failures],
        }

    async def _run(self) -> None:
        while not self._stopping:
            event = await self._queue.get()
            try:
                await self._handle(event)
            except asyncio.CancelledError:
                self._queue.put_nowait(event)
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
                self._record_failure(event, attempts, f"{type(exc).__name__}: {exc}")
                return

    def _record_failure(self, event: Event, attempts: int, error: str) -> None:
        memory_id = str(event.payload.get("memory_id", ""))
        failure = MaintenanceFailure(event.event_id, event.event_type.value, memory_id, attempts, error)
        failed_event = maintenance_event(
            EventType.MAINTENANCE_FAILED,
            memory_id=memory_id or "unknown",
            source="background_maintenance_supervisor",
            execution_id=event.execution_id,
            causation_id=event.event_id,
            payload={"attempts": attempts, "error": error},
        )
        self._unrecorded[event.event_id] = event, failure, failed_event
        self._publish_failure(event.event_id)

    def _publish_failure(self, event_id: str) -> None:
        event, failure, failed_event = self._unrecorded[event_id]
        if not any(item.event_id == event.event_id for item in self.failure_store.all()):
            self.failure_store.append(failure)
        if not any(
            item.event_id == failed_event.event_id
            for item in self.event_bus.store.filter(
                event_type=EventType.MAINTENANCE_FAILED, execution_id=event.execution_id
            )
        ):
            self.event_bus.store.append(failed_event)
        self._terminal_ids.add(event.event_id)
        self._pending_ids.discard(event.event_id)
        del self._unrecorded[event_id]

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
