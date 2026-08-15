"""Shared benchmark progress interface and ordered lifecycle tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from benchmarks.common.contracts import LifecycleStage, ProgressEvent, RunStatus


class ProgressSink(Protocol):
    """Consumer for benchmark lifecycle and per-item progress events."""

    def publish(self, event: ProgressEvent) -> None:
        """Receive one immutable event."""


@dataclass(slots=True)
class BenchmarkLifecycle:
    """Enforce forward-only lifecycle transitions and publish progress."""

    sink: ProgressSink | None = None
    _stage_index: int = field(default=-1, init=False)

    def transition(
        self,
        stage: LifecycleStage,
        *,
        current: int = 0,
        total: int = 0,
        case_id: str | None = None,
        message: str = "",
        status: RunStatus = RunStatus.PARTIAL,
    ) -> ProgressEvent:
        """Move forward or emit another event within the current stage."""

        order = tuple(LifecycleStage)
        index = order.index(stage)
        if index < self._stage_index:
            raise ValueError(f"benchmark lifecycle cannot move backward to {stage.value}")
        self._stage_index = index
        event = ProgressEvent(
            stage=stage,
            status=status,
            current=current,
            total=total,
            case_id=case_id,
            message=message,
        )
        if self.sink is not None:
            self.sink.publish(event)
        return event
