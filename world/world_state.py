"""Symbolic world-state snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


def clamp01(value: float) -> float:
    """Clamp a score into [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        return dict(data) if isinstance(data, dict) else {}
    return {}


@dataclass(frozen=True, slots=True)
class WorldState:
    """Immutable symbolic state consumed and produced by the world model."""

    goal_state: dict[str, Any] = field(default_factory=dict)
    task_state: dict[str, Any] = field(default_factory=dict)
    confidence_state: dict[str, Any] = field(default_factory=dict)
    emotional_state: dict[str, Any] = field(default_factory=dict)
    environment_state: dict[str, Any] = field(default_factory=dict)
    execution_state: dict[str, Any] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraints", tuple(str(item) for item in self.constraints))

    @classmethod
    def from_cognitive_state(
        cls,
        cognitive_state: Any | None,
        execution_state: dict[str, Any] | None = None,
        constraints: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> WorldState:
        """Create a world-state snapshot from a cognitive-state-like object."""
        return cls(
            goal_state=_as_dict(getattr(cognitive_state, "goal_state", {})),
            task_state=_as_dict(getattr(cognitive_state, "task_state", {})),
            confidence_state=_as_dict(getattr(cognitive_state, "confidence_state", {})),
            emotional_state=_as_dict(getattr(cognitive_state, "emotional_state", None)),
            environment_state=_as_dict(getattr(cognitive_state, "environment_state", {})),
            execution_state=execution_state or {},
            constraints=constraints,
            metadata=metadata or {},
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> WorldState:
        """Create a world-state snapshot from plain Python values."""
        data = data or {}
        return cls(
            goal_state=dict(data.get("goal_state", {})),
            task_state=dict(data.get("task_state", {})),
            confidence_state=dict(data.get("confidence_state", {})),
            emotional_state=dict(data.get("emotional_state", {})),
            environment_state=dict(data.get("environment_state", {})),
            execution_state=dict(data.get("execution_state", {})),
            constraints=tuple(str(item) for item in data.get("constraints", ())),
            metadata=dict(data.get("metadata", {})),
        )

    def with_updates(
        self,
        goal_state: dict[str, Any] | None = None,
        task_state: dict[str, Any] | None = None,
        confidence_state: dict[str, Any] | None = None,
        emotional_state: dict[str, Any] | None = None,
        environment_state: dict[str, Any] | None = None,
        execution_state: dict[str, Any] | None = None,
        constraints: tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorldState:
        """Return a copy with selected sections replaced."""
        return replace(
            self,
            goal_state=self.goal_state if goal_state is None else goal_state,
            task_state=self.task_state if task_state is None else task_state,
            confidence_state=self.confidence_state if confidence_state is None else confidence_state,
            emotional_state=self.emotional_state if emotional_state is None else emotional_state,
            environment_state=self.environment_state if environment_state is None else environment_state,
            execution_state=self.execution_state if execution_state is None else execution_state,
            constraints=self.constraints if constraints is None else constraints,
            metadata=self.metadata if metadata is None else metadata,
        )

    def merge_transition(self, transition: dict[str, Any]) -> WorldState:
        """Apply a symbolic transition dictionary and return the predicted state."""
        goal_state = dict(self.goal_state)
        goal_state.update(dict(transition.get("goal_state", {})))
        task_state = dict(self.task_state)
        task_state.update(dict(transition.get("task_state", {})))
        confidence_state = dict(self.confidence_state)
        confidence_state.update(dict(transition.get("confidence_state", {})))
        emotional_state = dict(self.emotional_state)
        emotional_state.update(dict(transition.get("emotional_state", {})))
        environment_state = dict(self.environment_state)
        environment_state.update(dict(transition.get("environment_state", {})))
        execution_state = dict(self.execution_state)
        execution_state.update(dict(transition.get("execution_state", {})))
        metadata = dict(self.metadata)
        metadata.update(dict(transition.get("metadata", {})))
        constraints = tuple(dict.fromkeys((*self.constraints, *tuple(transition.get("constraints", ()))))) 
        return self.with_updates(
            goal_state=goal_state,
            task_state=task_state,
            confidence_state=confidence_state,
            emotional_state=emotional_state,
            environment_state=environment_state,
            execution_state=execution_state,
            constraints=constraints,
            metadata=metadata,
        )

    def confidence(self) -> float:
        """Return the best symbolic confidence value known in this state."""
        value = self.confidence_state.get("overall_confidence", self.confidence_state.get("confidence", 0.0))
        try:
            return clamp01(float(value))
        except (TypeError, ValueError):
            return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the world state into plain Python values."""
        return {
            "goal_state": dict(self.goal_state),
            "task_state": dict(self.task_state),
            "confidence_state": dict(self.confidence_state),
            "emotional_state": dict(self.emotional_state),
            "environment_state": dict(self.environment_state),
            "execution_state": dict(self.execution_state),
            "constraints": list(self.constraints),
            "metadata": dict(self.metadata),
        }
