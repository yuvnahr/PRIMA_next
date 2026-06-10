"""Aggregation logic for unified confidence estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass

from uncertainty.calibration import ConfidenceCalibrator
from uncertainty.confidence_signal import ConfidenceSignal, DecisionProbability, OverallConfidence, clamp01
from uncertainty.uncertainty_types import ConfidenceSource, ConfidenceTrend, DecisionType, UncertaintyBand


@dataclass(frozen=True, slots=True)
class AggregationPolicy:
    """Weights and thresholds for probabilistic confidence aggregation."""

    reflect_uncertainty_midpoint: float = 0.45
    retry_uncertainty_midpoint: float = 0.6
    ask_user_uncertainty_midpoint: float = 0.75
    continue_confidence_midpoint: float = 0.55
    decision_temperature: float = 0.16


class ConfidenceAggregator:
    """Aggregate calibrated subsystem confidence into probabilistic decisions."""

    DEFAULT_WEIGHTS: dict[ConfidenceSource, float] = {
        ConfidenceSource.RETRIEVAL: 1.25,
        ConfidenceSource.REFLECTION: 1.0,
        ConfidenceSource.AFFECT: 0.7,
        ConfidenceSource.PLANNER: 1.15,
        ConfidenceSource.STATE: 0.85,
        ConfidenceSource.POLICY: 1.0,
    }

    def __init__(
        self,
        calibrator: ConfidenceCalibrator | None = None,
        policy: AggregationPolicy | None = None,
        source_weights: dict[ConfidenceSource, float] | None = None,
    ) -> None:
        self.calibrator = calibrator or ConfidenceCalibrator()
        self.policy = policy or AggregationPolicy()
        self.source_weights = source_weights or dict(self.DEFAULT_WEIGHTS)

    def aggregate(self, signals: tuple[ConfidenceSignal, ...]) -> OverallConfidence:
        """Aggregate subsystem signals into an OverallConfidence estimate."""
        weighted_signals = tuple(self._apply_source_weight(signal) for signal in signals)
        calibrated = self.calibrator.calibrate_many(weighted_signals)
        if not calibrated:
            return self._empty_estimate()

        total_weight = sum(signal.effective_weight for signal in calibrated)
        if total_weight <= 0:
            return self._empty_estimate(signals=calibrated)

        confidence = sum(signal.confidence * signal.effective_weight for signal in calibrated) / total_weight
        uncertainty = sum(signal.uncertainty * signal.effective_weight for signal in calibrated) / total_weight
        disagreement = self._weighted_disagreement(calibrated, confidence, total_weight)
        uncertainty = clamp01(uncertainty * 0.75 + disagreement * 0.25)
        confidence = clamp01(confidence * (1.0 - disagreement * 0.15))
        interval = self._confidence_interval(confidence, uncertainty, calibrated)
        decisions = self._decision_probabilities(confidence, uncertainty, calibrated)
        recommended = max(decisions, key=lambda decision: decision.probability).decision_type

        return OverallConfidence(
            confidence=round(confidence, 6),
            uncertainty=round(uncertainty, 6),
            confidence_interval=interval,
            uncertainty_band=self._band(uncertainty),
            decision_probabilities=decisions,
            recommended_decision=recommended,
            signals=calibrated,
            calibration={"profiles": self.calibrator.profile_snapshot()},
            metadata={
                "signal_count": len(calibrated),
                "total_effective_weight": round(total_weight, 6),
                "disagreement": round(disagreement, 6),
                "binary_confidence": False,
            },
        )

    def _apply_source_weight(self, signal: ConfidenceSignal) -> ConfidenceSignal:
        source_weight = self.source_weights.get(signal.source, 1.0)
        return ConfidenceSignal(
            source=signal.source,
            confidence=signal.confidence,
            uncertainty=signal.uncertainty,
            weight=signal.weight * source_weight,
            reliability=signal.reliability,
            evidence_count=signal.evidence_count,
            trend=signal.trend,
            metadata=dict(signal.metadata),
        )

    def _empty_estimate(self, signals: tuple[ConfidenceSignal, ...] = ()) -> OverallConfidence:
        decisions = (
            DecisionProbability(DecisionType.CONTINUE, 0.1, "No confidence signals are available."),
            DecisionProbability(DecisionType.REFLECT, 0.35, "Missing signals increase reflection value."),
            DecisionProbability(DecisionType.RETRY, 0.25, "Missing signals may indicate retryable uncertainty."),
            DecisionProbability(DecisionType.ASK_USER, 0.3, "Missing signals may require user clarification."),
        )
        return OverallConfidence(
            confidence=0.0,
            uncertainty=1.0,
            confidence_interval=(0.0, 0.25),
            uncertainty_band=UncertaintyBand.VERY_HIGH,
            decision_probabilities=decisions,
            recommended_decision=DecisionType.REFLECT,
            signals=signals,
            metadata={"signal_count": len(signals), "binary_confidence": False},
        )

    def _weighted_disagreement(
        self,
        signals: tuple[ConfidenceSignal, ...],
        mean_confidence: float,
        total_weight: float,
    ) -> float:
        variance = sum(
            ((signal.confidence - mean_confidence) ** 2) * signal.effective_weight
            for signal in signals
        ) / total_weight
        return clamp01(math.sqrt(variance) * 2.0)

    def _confidence_interval(
        self,
        confidence: float,
        uncertainty: float,
        signals: tuple[ConfidenceSignal, ...],
    ) -> tuple[float, float]:
        evidence = sum(max(1, signal.evidence_count) for signal in signals)
        width = clamp01((uncertainty * 0.45) + (1.0 / max(2.0, math.sqrt(evidence))) * 0.2)
        return (round(clamp01(confidence - width), 6), round(clamp01(confidence + width), 6))

    def _decision_probabilities(
        self,
        confidence: float,
        uncertainty: float,
        signals: tuple[ConfidenceSignal, ...],
    ) -> tuple[DecisionProbability, ...]:
        reflection_pressure = self._source_uncertainty(signals, ConfidenceSource.REFLECTION)
        retrieval_pressure = self._source_uncertainty(signals, ConfidenceSource.RETRIEVAL)
        planner_pressure = self._source_uncertainty(signals, ConfidenceSource.PLANNER)
        affect_pressure = self._source_uncertainty(signals, ConfidenceSource.AFFECT)
        degrading_pressure = self._trend_pressure(signals)

        continue_probability = self._sigmoid(confidence - self.policy.continue_confidence_midpoint)
        reflect_probability = self._sigmoid(
            uncertainty + reflection_pressure * 0.25 + affect_pressure * 0.15
            - self.policy.reflect_uncertainty_midpoint
        )
        retry_probability = self._sigmoid(
            uncertainty + retrieval_pressure * 0.2 + planner_pressure * 0.2 + degrading_pressure * 0.15
            - self.policy.retry_uncertainty_midpoint
        )
        ask_user_probability = self._sigmoid(
            uncertainty + retrieval_pressure * 0.15 + planner_pressure * 0.1
            - self.policy.ask_user_uncertainty_midpoint
        )

        total = continue_probability + reflect_probability + retry_probability + ask_user_probability
        if total <= 0:
            total = 1.0
        return (
            DecisionProbability(
                DecisionType.CONTINUE,
                round(continue_probability / total, 6),
                "Confidence is sufficient relative to uncertainty.",
            ),
            DecisionProbability(
                DecisionType.REFLECT,
                round(reflect_probability / total, 6),
                "Reflection probability rises with uncertainty, affect pressure, and reflection pressure.",
            ),
            DecisionProbability(
                DecisionType.RETRY,
                round(retry_probability / total, 6),
                "Retry probability rises with retrieval or planner uncertainty and degrading confidence.",
            ),
            DecisionProbability(
                DecisionType.ASK_USER,
                round(ask_user_probability / total, 6),
                "User clarification probability rises when uncertainty remains high after subsystem evidence.",
            ),
        )

    def _source_uncertainty(self, signals: tuple[ConfidenceSignal, ...], source: ConfidenceSource) -> float:
        matching = [signal for signal in signals if signal.source == source]
        if not matching:
            return 0.0
        total_weight = sum(signal.effective_weight for signal in matching)
        if total_weight <= 0:
            return 0.0
        return clamp01(sum(signal.uncertainty * signal.effective_weight for signal in matching) / total_weight)

    def _trend_pressure(self, signals: tuple[ConfidenceSignal, ...]) -> float:
        if not signals:
            return 0.0
        degrading = sum(1 for signal in signals if signal.trend == ConfidenceTrend.DEGRADING)
        improving = sum(1 for signal in signals if signal.trend == ConfidenceTrend.IMPROVING)
        return clamp01((degrading - improving * 0.5) / len(signals))

    def _sigmoid(self, value: float) -> float:
        scaled = value / max(0.01, self.policy.decision_temperature)
        return 1.0 / (1.0 + math.exp(-scaled))

    def _band(self, uncertainty: float) -> UncertaintyBand:
        if uncertainty < 0.2:
            return UncertaintyBand.VERY_LOW
        if uncertainty < 0.4:
            return UncertaintyBand.LOW
        if uncertainty < 0.6:
            return UncertaintyBand.MODERATE
        if uncertainty < 0.8:
            return UncertaintyBand.HIGH
        return UncertaintyBand.VERY_HIGH
