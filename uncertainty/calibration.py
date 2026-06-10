"""Calibration helpers for probabilistic confidence scores."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from uncertainty.confidence_signal import ConfidenceSignal, clamp01
from uncertainty.uncertainty_types import ConfidenceSource


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """Per-source calibration settings.

    Temperature controls how strongly scores move toward extremes. Bias shifts
    confidence before temperature scaling. Reliability is an explicit prior for
    the source and is still combined with each signal's own reliability.
    """

    source: ConfidenceSource
    temperature: float = 1.0
    bias: float = 0.0
    reliability: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", ConfidenceSource(self.source))
        object.__setattr__(self, "temperature", max(0.05, float(self.temperature)))
        object.__setattr__(self, "bias", max(-1.0, min(1.0, float(self.bias))))
        object.__setattr__(self, "reliability", clamp01(self.reliability))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the calibration profile into plain Python values."""
        return {
            "source": self.source.value,
            "temperature": self.temperature,
            "bias": self.bias,
            "reliability": self.reliability,
            "metadata": dict(self.metadata),
        }


class ConfidenceCalibrator:
    """Apply deterministic calibration profiles to confidence signals."""

    def __init__(self, profiles: dict[ConfidenceSource, CalibrationProfile] | None = None) -> None:
        self.profiles = profiles or self.default_profiles()

    @staticmethod
    def default_profiles() -> dict[ConfidenceSource, CalibrationProfile]:
        """Return conservative default calibration settings."""
        return {
            ConfidenceSource.RETRIEVAL: CalibrationProfile(ConfidenceSource.RETRIEVAL, temperature=1.0, reliability=0.95),
            ConfidenceSource.REFLECTION: CalibrationProfile(
                ConfidenceSource.REFLECTION,
                temperature=1.1,
                reliability=0.9,
            ),
            ConfidenceSource.AFFECT: CalibrationProfile(ConfidenceSource.AFFECT, temperature=1.2, reliability=0.75),
            ConfidenceSource.PLANNER: CalibrationProfile(ConfidenceSource.PLANNER, temperature=1.0, reliability=0.9),
            ConfidenceSource.STATE: CalibrationProfile(ConfidenceSource.STATE, temperature=1.0, reliability=0.8),
            ConfidenceSource.POLICY: CalibrationProfile(ConfidenceSource.POLICY, temperature=0.9, reliability=1.0),
        }

    def calibrate_signal(self, signal: ConfidenceSignal) -> ConfidenceSignal:
        """Return a calibrated signal without mutating the input."""
        profile = self.profiles.get(signal.source, CalibrationProfile(signal.source))
        confidence = self._temperature_scale(signal.confidence, profile.temperature, profile.bias)
        uncertainty = clamp01((signal.uncertainty * 0.7) + ((1.0 - profile.reliability) * 0.3))
        reliability = clamp01(signal.reliability * profile.reliability)
        metadata = dict(signal.metadata)
        metadata["calibration_profile"] = profile.to_dict()
        calibrated = ConfidenceSignal(
            source=signal.source,
            confidence=confidence,
            uncertainty=uncertainty,
            weight=signal.weight,
            reliability=reliability,
            evidence_count=signal.evidence_count,
            trend=signal.trend,
            metadata=metadata,
        )
        return calibrated

    def calibrate_many(self, signals: tuple[ConfidenceSignal, ...]) -> tuple[ConfidenceSignal, ...]:
        """Calibrate a tuple of signals."""
        return tuple(self.calibrate_signal(signal) for signal in signals)

    def profile_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return profiles keyed by source value."""
        return {source.value: profile.to_dict() for source, profile in self.profiles.items()}

    def _temperature_scale(self, confidence: float, temperature: float, bias: float) -> float:
        centered = clamp01(confidence + bias)
        if centered <= 0.0 or centered >= 1.0:
            return centered
        logit = math.log(centered / (1.0 - centered))
        scaled = logit / temperature
        return round(clamp01(1.0 / (1.0 + math.exp(-scaled))), 6)
