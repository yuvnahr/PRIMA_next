"""Prediction result models for the symbolic world model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from world.world_state import WorldState, clamp01


@dataclass(frozen=True, slots=True)
class StateTransition:
    """Predicted symbolic state transition."""

    action_id: str
    before_state: WorldState
    after_state: WorldState
    delta: dict[str, Any]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the transition into plain Python values."""
        return {
            "action_id": self.action_id,
            "before_state": self.before_state.to_dict(),
            "after_state": self.after_state.to_dict(),
            "delta": dict(self.delta),
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class ConstraintPrediction:
    """Predicted constraint pressure for a plan or action."""

    constraint_id: str
    description: str
    violation_probability: float
    severity: float
    mitigation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "violation_probability", clamp01(self.violation_probability))
        object.__setattr__(self, "severity", clamp01(self.severity))

    @property
    def risk_contribution(self) -> float:
        """Return the weighted contribution this constraint makes to risk."""
        return clamp01(self.violation_probability * self.severity)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the constraint prediction into plain Python values."""
        return {
            "constraint_id": self.constraint_id,
            "description": self.description,
            "violation_probability": self.violation_probability,
            "severity": self.severity,
            "risk_contribution": self.risk_contribution,
            "mitigation": self.mitigation,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ActionPrediction:
    """Predicted outcome for one symbolic plan action."""

    action_id: str
    action_type: str
    expected_state: WorldState
    risk_score: float
    failure_probability: float
    constraint_predictions: tuple[ConstraintPrediction, ...] = ()
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "risk_score", clamp01(self.risk_score))
        object.__setattr__(self, "failure_probability", clamp01(self.failure_probability))
        object.__setattr__(self, "constraint_predictions", tuple(self.constraint_predictions))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the action prediction into plain Python values."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "expected_state": self.expected_state.to_dict(),
            "risk_score": self.risk_score,
            "failure_probability": self.failure_probability,
            "constraint_predictions": [item.to_dict() for item in self.constraint_predictions],
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Aggregate symbolic prediction for a plan."""

    initial_state: WorldState
    expected_state: WorldState
    action_predictions: tuple[ActionPrediction, ...]
    state_transitions: tuple[StateTransition, ...]
    constraint_predictions: tuple[ConstraintPrediction, ...]
    risk_score: float
    failure_probability: float
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_predictions", tuple(self.action_predictions))
        object.__setattr__(self, "state_transitions", tuple(self.state_transitions))
        object.__setattr__(self, "constraint_predictions", tuple(self.constraint_predictions))
        object.__setattr__(self, "risk_score", clamp01(self.risk_score))
        object.__setattr__(self, "failure_probability", clamp01(self.failure_probability))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prediction result into plain Python values."""
        return {
            "initial_state": self.initial_state.to_dict(),
            "expected_state": self.expected_state.to_dict(),
            "action_predictions": [item.to_dict() for item in self.action_predictions],
            "state_transitions": [item.to_dict() for item in self.state_transitions],
            "constraint_predictions": [item.to_dict() for item in self.constraint_predictions],
            "risk_score": self.risk_score,
            "failure_probability": self.failure_probability,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }
