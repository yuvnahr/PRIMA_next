"""Unified uncertainty estimator for PRIMA-NEXT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from uncertainty.confidence_aggregator import ConfidenceAggregator
from uncertainty.confidence_signal import ConfidenceSignal, OverallConfidence, clamp01
from uncertainty.uncertainty_types import ConfidenceSource, ConfidenceTrend


@dataclass(slots=True)
class UncertaintyEstimator:
    """Build confidence signals and aggregate probabilistic workflow guidance."""

    aggregator: ConfidenceAggregator = field(default_factory=ConfidenceAggregator)

    def estimate(
        self,
        retrieval_confidence: Any | None = None,
        reflection_confidence: Any | None = None,
        affect_confidence: Any | None = None,
        planner_confidence: Any | None = None,
        extra_signals: tuple[ConfidenceSignal, ...] = (),
    ) -> OverallConfidence:
        """Estimate overall confidence from subsystem confidence objects."""
        signals = self.signals_from_inputs(
            retrieval_confidence=retrieval_confidence,
            reflection_confidence=reflection_confidence,
            affect_confidence=affect_confidence,
            planner_confidence=planner_confidence,
            extra_signals=extra_signals,
        )
        return self.aggregator.aggregate(signals)

    async def estimate_async(
        self,
        retrieval_confidence: Any | None = None,
        reflection_confidence: Any | None = None,
        affect_confidence: Any | None = None,
        planner_confidence: Any | None = None,
        extra_signals: tuple[ConfidenceSignal, ...] = (),
    ) -> OverallConfidence:
        """Async-compatible wrapper around estimate."""
        return self.estimate(
            retrieval_confidence=retrieval_confidence,
            reflection_confidence=reflection_confidence,
            affect_confidence=affect_confidence,
            planner_confidence=planner_confidence,
            extra_signals=extra_signals,
        )

    def estimate_from_subsystem_outputs(
        self,
        retrieval_response: Any | None = None,
        reflection_result: Any | None = None,
        affect_update: Any | None = None,
        plan: Any | None = None,
        extra_signals: tuple[ConfidenceSignal, ...] = (),
    ) -> OverallConfidence:
        """Estimate confidence from workflow-routed subsystem outputs."""
        return self.estimate(
            retrieval_confidence=getattr(retrieval_response, "confidence", retrieval_response),
            reflection_confidence=getattr(reflection_result, "confidence", reflection_result),
            affect_confidence=affect_update,
            planner_confidence=plan,
            extra_signals=extra_signals,
        )

    def signals_from_inputs(
        self,
        retrieval_confidence: Any | None = None,
        reflection_confidence: Any | None = None,
        affect_confidence: Any | None = None,
        planner_confidence: Any | None = None,
        extra_signals: tuple[ConfidenceSignal, ...] = (),
    ) -> tuple[ConfidenceSignal, ...]:
        """Normalize supported subsystem confidence objects into signals."""
        signals: list[ConfidenceSignal] = []
        retrieval_signal = self._retrieval_signal(retrieval_confidence)
        reflection_signal = self._reflection_signal(reflection_confidence)
        affect_signal = self._affect_signal(affect_confidence)
        planner_signal = self._planner_signal(planner_confidence)

        for signal in (retrieval_signal, reflection_signal, affect_signal, planner_signal):
            if signal is not None:
                signals.append(signal)
        signals.extend(extra_signals)
        return tuple(signals)

    def _retrieval_signal(self, value: Any | None) -> ConfidenceSignal | None:
        if value is None:
            return None
        confidence = _float_attr(value, "confidence", default=None)
        if confidence is None:
            return None
        # ensure defaults are computed with concrete floats
        default_ambiguity = 1.0 - confidence
        ambiguity = _float_attr(value, "ambiguity_score", default=default_ambiguity)
        if ambiguity is None:
            ambiguity = default_ambiguity
        coverage = _float_attr(value, "coverage_score", default=confidence)
        if coverage is None:
            coverage = confidence
        quality = _float_attr(value, "retrieval_quality_score", default=confidence)
        if quality is None:
            quality = confidence
        uncertainty = clamp01((1.0 - confidence) * 0.45 + ambiguity * 0.35 + (1.0 - coverage) * 0.2)
        return ConfidenceSignal(
            source=ConfidenceSource.RETRIEVAL,
            confidence=confidence,
            uncertainty=uncertainty,
            reliability=quality,
            evidence_count=max(1, round(coverage * 5)),
            metadata={
                "ambiguity_score": ambiguity,
                "coverage_score": coverage,
                "retrieval_quality_score": quality,
            },
        )

    def _reflection_signal(self, value: Any | None) -> ConfidenceSignal | None:
        if value is None:
            return None
        confidence = _float_attr(value, "overall_confidence", default=None)
        if confidence is None:
            confidence = _float_attr(value, "confidence", default=None)
        if confidence is None:
            return None
        evidence = _float_attr(value, "evidence_strength", default=confidence)
        if evidence is None:
            evidence = confidence
        relevance = _float_attr(value, "reflection_relevance", default=confidence)
        if relevance is None:
            relevance = confidence
        retrieval_support = _float_attr(value, "retrieval_support", default=confidence)
        if retrieval_support is None:
            retrieval_support = confidence
        state_support = _float_attr(value, "state_support", default=confidence)
        if state_support is None:
            state_support = confidence
        support_gap = 1.0 - ((evidence + relevance + retrieval_support + state_support) / 4.0)
        uncertainty = clamp01((1.0 - confidence) * 0.6 + support_gap * 0.4)
        return ConfidenceSignal(
            source=ConfidenceSource.REFLECTION,
            confidence=confidence,
            uncertainty=uncertainty,
            reliability=clamp01((evidence + relevance) / 2.0),
            evidence_count=4,
            metadata={
                "evidence_strength": evidence,
                "reflection_relevance": relevance,
                "retrieval_support": retrieval_support,
                "state_support": state_support,
            },
        )

    def _affect_signal(self, value: Any | None) -> ConfidenceSignal | None:
        if value is None:
            return None
        profile = getattr(value, "profile", value)
        confidence = _float_attr(profile, "confidence", default=None)
        if confidence is None:
            confidence = _float_attr(value, "confidence", default=None)
        if confidence is None:
            return None
        dissonance = _float_attr(value, "dissonance_score", default=0.0)
        if dissonance is None:
            dissonance = 0.0
        salience = _float_attr(value, "salience_score", default=0.0)
        if salience is None:
            salience = 0.0
        volatility = _float_from_mapping(getattr(value, "evolution", {}), "volatility", default=0.0)
        uncertainty = clamp01((1.0 - confidence) * 0.45 + dissonance * 0.35 + volatility * 0.15 + salience * 0.05)
        trend = ConfidenceTrend.DEGRADING if dissonance >= 0.6 or volatility >= 0.6 else ConfidenceTrend.STABLE
        return ConfidenceSignal(
            source=ConfidenceSource.AFFECT,
            confidence=confidence,
            uncertainty=uncertainty,
            reliability=clamp01(1.0 - dissonance * 0.4),
            evidence_count=1,
            trend=trend,
            metadata={
                "dominant_emotion": getattr(profile, "dominant_emotion", "neutral"),
                "dissonance_score": dissonance,
                "salience_score": salience,
                "volatility": volatility,
            },
        )

    def _planner_signal(self, value: Any | None) -> ConfidenceSignal | None:
        if value is None:
            return None
        evaluation = getattr(value, "evaluation", value)
        intent = getattr(value, "execution_intent", None)
        confidence = _float_attr(evaluation, "confidence", default=None)
        intent_confidence = _float_attr(intent, "confidence", default=None)
        if confidence is None:
            confidence = intent_confidence
        elif intent_confidence is not None:
            confidence = clamp01(confidence * 0.75 + intent_confidence * 0.25)
        if confidence is None:
            return None

        default_uncertainty = 1.0 - confidence
        uncertainty = _float_attr(evaluation, "uncertainty", default=default_uncertainty)
        if uncertainty is None:
            uncertainty = default_uncertainty
        risk = _float_attr(evaluation, "risk_score", default=uncertainty)
        if risk is None:
            risk = uncertainty
        should_replan = bool(getattr(evaluation, "should_replan", False))
        if should_replan:
            uncertainty = clamp01(uncertainty + 0.2)
        trend = ConfidenceTrend.DEGRADING if should_replan or risk >= 0.65 else ConfidenceTrend.STABLE
        reasons = tuple(getattr(evaluation, "reasons", ()) or ())
        return ConfidenceSignal(
            source=ConfidenceSource.PLANNER,
            confidence=confidence,
            uncertainty=clamp01(uncertainty * 0.75 + risk * 0.25),
            reliability=clamp01(1.0 - risk * 0.35),
            evidence_count=max(1, len(reasons) + 1),
            trend=trend,
            metadata={
                "risk_score": risk,
                "should_replan": should_replan,
                "reasons": list(reasons),
            },
        )


def _float_attr(value: Any, name: str, default: float | None) -> float | None:
    if value is None:
        return default
    raw = getattr(value, name, None)
    if raw is None and isinstance(value, dict):
        raw = value.get(name)
    if raw is None:
        return default
    try:
        return clamp01(float(raw))
    except (TypeError, ValueError):
        return default


def _float_from_mapping(value: Any, key: str, default: float) -> float:
    if not isinstance(value, dict):
        return default
    try:
        return clamp01(float(value.get(key, default)))
    except (TypeError, ValueError):
        return default
