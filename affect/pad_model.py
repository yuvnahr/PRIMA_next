"""Pleasure-arousal-dominance coordinate model."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    """Clamp a PAD dimension to the normalized model range."""
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True, slots=True)
class PADState:
    """Normalized pleasure-arousal-dominance coordinates.

    All dimensions are constrained to [-1, 1]. Methods return new immutable
    states, which keeps temporal updates deterministic and easy to audit.
    """

    pleasure: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pleasure", clamp(self.pleasure))
        object.__setattr__(self, "arousal", clamp(self.arousal))
        object.__setattr__(self, "dominance", clamp(self.dominance))

    def update(self, delta: PADState, rate: float = 1.0) -> PADState:
        """Apply a scaled delta to the current state."""
        return PADState(
            self.pleasure + delta.pleasure * rate,
            self.arousal + delta.arousal * rate,
            self.dominance + delta.dominance * rate,
        )

    def decay(self, factor: float = 0.92, baseline: PADState | None = None) -> PADState:
        """Decay this state toward a baseline."""
        base = baseline or PADState()
        return PADState(
            base.pleasure + (self.pleasure - base.pleasure) * factor,
            base.arousal + (self.arousal - base.arousal) * factor,
            base.dominance + (self.dominance - base.dominance) * factor,
        )

    def distance(self, other: PADState) -> float:
        """Euclidean distance between two PAD states."""
        return sqrt(
            (self.pleasure - other.pleasure) ** 2
            + (self.arousal - other.arousal) ** 2
            + (self.dominance - other.dominance) ** 2
        )

    def merge(self, other: PADState, weight: float = 0.5) -> PADState:
        """Blend this state with another state."""
        bounded_weight = max(0.0, min(1.0, weight))
        inverse = 1.0 - bounded_weight
        return PADState(
            self.pleasure * inverse + other.pleasure * bounded_weight,
            self.arousal * inverse + other.arousal * bounded_weight,
            self.dominance * inverse + other.dominance * bounded_weight,
        )

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.pleasure, self.arousal, self.dominance)

    def to_dict(self) -> dict[str, float]:
        return {
            "pleasure": self.pleasure,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> PADState:
        return cls(
            pleasure=data.get("pleasure", 0.0),
            arousal=data.get("arousal", 0.0),
            dominance=data.get("dominance", 0.0),
        )
