"""Observe foreground workflow events without owning execution or cognitive results."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from demo_cli.renderer import Renderer
from events import Event, EventBus, EventType
from workflow.workflow_events import WORKFLOW_EVENT_TYPE_MAP

MAX_LIVE_EVENTS = 50


class LiveTraceObserver:
    def __init__(self, event_bus: EventBus, renderer: Renderer) -> None:
        self.event_bus = event_bus
        self.renderer = renderer
        self.closed = False
        self._task: asyncio.Task[Any] | None = None
        self._execution_id: str | None = None
        self._opened = False
        self._displayed = 0
        self._suppressed = False
        self._subscriber = event_bus.subscribe(
            self.handle_event, event_types=tuple(WORKFLOW_EVENT_TYPE_MAP.values()),
            topics=("workflow",), sources=("workflow",), name="demo_live_trace",
        )

    def begin_turn(self, *, enabled: bool) -> None:
        self._task = asyncio.current_task() if enabled and not self.closed else None
        self._execution_id = None
        self._opened = False
        self._displayed = 0
        self._suppressed = False

    def handle_event(self, event: Event) -> None:
        # The orchestrator awaits publication on the foreground task; off-task events never print.
        try:
            task = asyncio.current_task()
        except RuntimeError:
            return
        if self.closed or self._task is None or task is not self._task:
            return
        try:
            self._render_event(event)
        except Exception:
            self._task = None
            try:
                self.renderer.message("Live workflow trace unavailable; conversation execution continues.")
            except Exception:
                return

    def _render_event(self, event: Event) -> None:
        kind = event.event_type
        if kind is EventType.WORKFLOW_STARTED:
            if self._execution_id is None and event.execution_id:
                self._execution_id = event.execution_id
                self._opened = True
                self.renderer.live_start()
            return
        if self._execution_id is None or event.execution_id != self._execution_id:
            return
        detail = ""
        if kind in {EventType.WORKFLOW_FAILED, EventType.WORKFLOW_CANCELLED}:
            marker, phase = "[!!]", "workflow"
            detail = "failed" if kind is EventType.WORKFLOW_FAILED else "cancelled"
        else:
            phase = event.payload.get("phase")
            if not isinstance(phase, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", phase):
                return
            if kind is EventType.WORKFLOW_PHASE_STARTED and phase == "answer_generation":
                marker = "[>>]"
            elif kind is EventType.WORKFLOW_PHASE_COMPLETED:
                marker = "[OK]"
            elif kind is EventType.WORKFLOW_PHASE_RETRY:
                marker, detail = "[!!]", "retry"
                count = event.payload.get("retry_count")
                if type(count) is int and 0 < count < 10000:
                    detail += f" {count}"
            elif kind is EventType.WORKFLOW_PHASE_FAILED:
                marker, detail = "[!!]", "failed"
            else:
                return
        if self._displayed >= MAX_LIVE_EVENTS:
            if not self._suppressed:
                self._suppressed = True
                self.renderer.message("... additional workflow events suppressed")
            return
        self._displayed += 1
        self.renderer.live_event(marker, phase, detail)

    def end_turn(self) -> None:
        self._task = None
        self._execution_id = None
        opened, self._opened = self._opened, False
        if opened:
            try:
                self.renderer.live_end()
            except Exception:
                return

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.end_turn()
            self.event_bus.unsubscribe(self._subscriber)
