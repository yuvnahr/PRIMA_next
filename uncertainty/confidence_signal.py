"""Confidence signal and overall confidence domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from uncertainty.uncertainty_types import (
    ConfidenceSource,
    ConfidenceTrend,
    DecisionType,
    UncertaintyBand,
)


def clamp01(value: float) -> float:
    """Clamp a probabilistic score into [0, 1]."""
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class ConfidenceSignal:
    """A normalized confidence contribution from one subsystem."""

    source: ConfidenceSource
    confidence: float
    uncertainty: float
    weight: float = 1.0
    reliability: float = 1.0
    evidence_count: int = 1
    trend: ConfidenceTrend = ConfidenceTrend.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", ConfidenceSource(self.source))
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "uncertainty", clamp01(self.uncertainty))
        object.__setattr__(self, "weight", max(0.0, float(self.weight)))
        object.__setattr__(self, "reliability", clamp01(self.reliability))
        object.__setattr__(self, "evidence_count", max(0, int(self.evidence_count)))
        object.__setattr__(self, "trend", ConfidenceTrend(self.trend))

    @property
    def effective_weight(self) -> float:
        """Return the reliability-adjusted aggregation weight."""
        evidence_factor = min(1.0, self.evidence_count / 5.0) if self.evidence_count else 0.25
        return self.weight * self.reliability * max(0.25, evidence_factor)

    def calibrated(self, confidence: float, uncertainty: float, reliability: float | None = None) -> ConfidenceSignal:
        """Return a calibrated copy of this signal."""
        return ConfidenceSignal(
            source=self.source,
            confidence=confidence,
            uncertainty=uncertainty,
            weight=self.weight,
            reliability=self.reliability if reliability is None else reliability,
            evidence_count=self.evidence_count,
            trend=self.trend,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the signal into plain Python values."""
        return {
            "source": self.source.value,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "weight": self.weight,
            "reliability": self.reliability,
            "effective_weight": self.effective_weight,
            "evidence_count": self.evidence_count,
            "trend": self.trend.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DecisionProbability:
    """Continuous probability for one workflow decision."""

    decision_type: DecisionType
    probability: float
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_type", DecisionType(self.decision_type))
        object.__setattr__(self, "probability", clamp01(self.probability))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision probability into plain Python values."""
        return {
            "decision_type": self.decision_type.value,
            "probability": self.probability,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class OverallConfidence:
    """Unified probabilistic confidence estimate.

    This model intentionally avoids binary confidence. Workflow decisions are
    expressed as probabilities plus a recommended highest-probability action.
    """

    confidence: float
    uncertainty: float
    confidence_interval: tuple[float, float]
    uncertainty_band: UncertaintyBand
    decision_probabilities: tuple[DecisionProbability, ...]
    recommended_decision: DecisionType
    signals: tuple[ConfidenceSignal, ...]
    calibration: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "uncertainty", clamp01(self.uncertainty))
        lower, upper = self.confidence_interval
        object.__setattr__(self, "confidence_interval", (clamp01(lower), clamp01(upper)))
        object.__setattr__(self, "uncertainty_band", UncertaintyBand(self.uncertainty_band))
        object.__setattr__(self, "decision_probabilities", tuple(self.decision_probabilities))
        object.__setattr__(self, "recommended_decision", DecisionType(self.recommended_decision))
        object.__setattr__(self, "signals", tuple(self.signals))

    def probability_for(self, decision_type: DecisionType) -> float:
        """Return the probability assigned to a workflow decision."""
        requested = DecisionType(decision_type)
        for decision in self.decision_probabilities:
            if decision.decision_type == requested:
                return decision.probability
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the estimate into plain Python values."""
        return {
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "confidence_interval": list(self.confidence_interval),
            "uncertainty_band": self.uncertainty_band.value,
            "decision_probabilities": [decision.to_dict() for decision in self.decision_probabilities],
            "recommended_decision": self.recommended_decision.value,
            "signals": [signal.to_dict() for signal in self.signals],
            "calibration": dict(self.calibration),
            "metadata": dict(self.metadata),
        }
