"""Typed, versioned cognitive state shared by runtime task routes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from state.emotional_state import EmotionalState


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class StateSection(MutableMapping[str, Any]):
    """Named cognitive-state section with mapping compatibility."""

    data: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self.data == dict(other)
        return False


class GoalState(StateSection):
    """Goals and constraints."""


class TaskState(StateSection):
    """Current task attributes."""


class ConfidenceState(StateSection):
    """Confidence observations."""


class EnvironmentState(StateSection):
    """Persistable execution environment."""


@dataclass(slots=True, init=False)
class CognitiveState:
    """Versioned state aggregate owned by a state manager."""

    goal: GoalState
    emotional: EmotionalState
    task: TaskState
    confidence: ConfidenceState
    environment: EnvironmentState
    metadata: dict[str, Any]
    version: int
    created_at: datetime
    updated_at: datetime

    def __init__(
        self,
        goal: GoalState | Mapping[str, Any] | None = None,
        emotional: EmotionalState | None = None,
        task: TaskState | Mapping[str, Any] | None = None,
        confidence: ConfidenceState | Mapping[str, Any] | None = None,
        environment: EnvironmentState | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        version: int = 0,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        **legacy: Any,
    ) -> None:
        goal = legacy.pop("goal_state", goal)
        emotional = legacy.pop("emotional_state", emotional)
        task = legacy.pop("task_state", task)
        confidence = legacy.pop("confidence_state", confidence)
        environment = legacy.pop("environment_state", environment)
        if legacy:
            raise TypeError(f"Unexpected cognitive state fields: {', '.join(sorted(legacy))}")
        now = utc_now()
        self.goal = goal if isinstance(goal, GoalState) else GoalState(dict(goal or {}))
        self.emotional = emotional or EmotionalState()
        self.task = task if isinstance(task, TaskState) else TaskState(dict(task or {}))
        self.confidence = (
            confidence if isinstance(confidence, ConfidenceState) else ConfidenceState(dict(confidence or {}))
        )
        self.environment = (
            environment if isinstance(environment, EnvironmentState) else EnvironmentState(dict(environment or {}))
        )
        self.metadata, self.version = dict(metadata or {}), version
        self.created_at, self.updated_at = created_at or now, updated_at or now

    goal_state = property(lambda self: self.goal)
    emotional_state = property(lambda self: self.emotional, lambda self, value: setattr(self, "emotional", value))
    task_state = property(lambda self: self.task)
    confidence_state = property(lambda self: self.confidence)
    environment_state = property(lambda self: self.environment)

    def to_dict(self) -> dict[str, Any]:
        """Serialize every state section and ownership field."""
        return {
            "goal": dict(self.goal),
            "emotional": self.emotional.to_dict(),
            "task": dict(self.task),
            "confidence": dict(self.confidence),
            "environment": dict(self.environment),
            "metadata": dict(self.metadata),
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CognitiveState:
        """Load state from persisted JSON data."""
        emotional = value.get("emotional", value.get("emotional_state", {}))
        created = value.get("created_at")
        updated = value.get("updated_at")
        return cls(
            goal=value.get("goal", value.get("goal_state", {})),
            emotional=EmotionalState.from_dict(dict(emotional)) if isinstance(emotional, Mapping) else None,
            task=value.get("task", value.get("task_state", {})),
            confidence=value.get("confidence", value.get("confidence_state", {})),
            environment=value.get("environment", value.get("environment_state", {})),
            metadata=value.get("metadata", {}),
            version=int(value.get("version", 0)),
            created_at=datetime.fromisoformat(created) if isinstance(created, str) else None,
            updated_at=datetime.fromisoformat(updated) if isinstance(updated, str) else None,
        )
