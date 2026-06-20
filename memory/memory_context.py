"""State-aware memory context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    goal_state: dict[str, Any] = field(default_factory=dict)
    task_state: dict[str, Any] = field(default_factory=dict)
    confidence_state: dict[str, Any] = field(default_factory=dict)
    emotional_state: dict[str, Any] = field(default_factory=dict)
    environment_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_state": dict(self.goal_state),
            "task_state": dict(self.task_state),
            "confidence_state": dict(self.confidence_state),
            "emotional_state": dict(self.emotional_state),
            "environment_state": dict(self.environment_state),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> StateSnapshot:
        data = data or {}
        return cls(
            goal_state=dict(data.get("goal_state", {})),
            task_state=dict(data.get("task_state", {})),
            confidence_state=dict(data.get("confidence_state", {})),
            emotional_state=dict(data.get("emotional_state", {})),
            environment_state=dict(data.get("environment_state", {})),
        )
