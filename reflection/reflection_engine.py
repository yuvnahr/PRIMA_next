"""Adaptive reflection engine."""

from __future__ import annotations

import uuid

from memory.retrieval.retrieval_confidence import RetrievalConfidence
from reflection.failure_classifier import FailureClassifier
from reflection.reflection_confidence import ReflectionConfidenceEstimator
from reflection.reflection_context import ReflectionContext
from reflection.reflection_lineage import ReflectionLineage
from reflection.reflection_memory import ReflectionMemory
from reflection.reflection_repository import ReflectionRepository
from reflection.reflection_result import ReflectionResult
from reflection.reflection_signal import ReflectionSignal
from reflection.reflection_types import FailureType, ReflectionSignalType, ReflectionSource
from reflection.rule_extractor import RuleExtractor


from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReflectionTriggerWeights:
    affect_weight: float = 0.70
    retrieval_weight: float = 0.20
    contradiction_weight: float = 0.10

FAILURE_TO_SIGNAL = {
    FailureType.TOOL_FAILURE: ReflectionSignalType.TOOL_FAILURE,
    FailureType.MEMORY_FAILURE: ReflectionSignalType.MEMORY_CONFLICT,
    FailureType.RETRIEVAL_FAILURE: ReflectionSignalType.RETRIEVAL_AMBIGUITY,
    FailureType.PLANNING_FAILURE: ReflectionSignalType.PLANNING_FAILURE,
    FailureType.REASONING_FAILURE: ReflectionSignalType.REASONING_FAILURE,
    FailureType.HALLUCINATION_RISK: ReflectionSignalType.HALLUCINATION_RISK,
    FailureType.GOAL_CONFLICT: ReflectionSignalType.GOAL_CONFLICT,
    FailureType.STATE_CONFLICT: ReflectionSignalType.STATE_INCONSISTENCY,
    FailureType.EMOTIONAL_CONFLICT: ReflectionSignalType.EMOTIONAL_DISSONANCE,
}


class ReflectionEngine:
    """Dependency-injected reflection component.

    The engine emits structured outputs and does not call planner, retrieval,
    affect, or memory internals directly.
    """

    def __init__(
        self,
        reflection_repository: ReflectionRepository | None = None,
        failure_classifier: FailureClassifier | None = None,
        rule_extractor: RuleExtractor | None = None,
        confidence_estimator: ReflectionConfidenceEstimator | None = None,
        trigger_threshold: float = 0.52,
        trigger_weights: ReflectionTriggerWeights | None = None,
    ) -> None:
        self.reflection_repository = reflection_repository or ReflectionRepository()
        self.failure_classifier = failure_classifier or FailureClassifier()
        self.rule_extractor = rule_extractor or RuleExtractor()
        self.confidence_estimator = confidence_estimator or ReflectionConfidenceEstimator()
        self.trigger_threshold = trigger_threshold
        self.trigger_weights = trigger_weights or ReflectionTriggerWeights()

    def evaluate(self, context: ReflectionContext) -> ReflectionResult:
        failure_type = context.failure_type or self.failure_classifier.classify(
            str(context.failure_metadata.get("reason", "")),
            context.failure_metadata,
        )
        signals = self._build_signals(context, failure_type)
        trigger_score = self.compute_trigger_score(context, signals)
        trigger_reasons = self.trigger_reasons(context, signals, trigger_score, failure_type)
        should_reflect = bool(context.failure_metadata.get("reasoning_event")) or (trigger_score >= self.trigger_threshold and any(bool(reason.get("triggered", False)) for reason in trigger_reasons))
        before_confidence = self._before_confidence(context)

        reflection_text = ""
        reflection_memory = None
        rules = []
        if should_reflect:
            reflection_text = self.generate_reflection_text(context, failure_type)
            lineage = ReflectionLineage(
                source_failure_ids=tuple(str(context.failure_metadata.get("failure_id", f"failure_{uuid.uuid4()}")).split(",")),
                generation=len(context.reflection_history),
            )
            confidence = self.confidence_estimator.estimate(
                reflection_text,
                evidence_strength=self._evidence_strength(context),
                retrieval_support=self._retrieval_support(context.retrieval_confidence),
                state_support=self._state_support(context),
            )
            reflection_memory = ReflectionMemory(
                reflection_id=f"reflection_{uuid.uuid4()}",
                source_failure=failure_type.value,
                reflection_text=reflection_text,
                reflection_signal=signals[0],
                confidence=confidence.overall_confidence,
                lineage=lineage,
                retrieval_context={
                    "query": context.query,
                    "retrieved_memory_count": len(context.retrieved_memories),
                    "retrieval_confidence": self._retrieval_support(context.retrieval_confidence),
                },
                state_snapshot=context.state_snapshot(),
            )
            self.reflection_repository.save_reflection(reflection_memory)

            if context.failure_metadata.get("successful_trace"):
                rule = self.rule_extractor.extract_rule(
                    context.query,
                    str(context.failure_metadata["successful_trace"]),
                    source_failures=lineage.source_failure_ids,
                    lineage=lineage,
                )
                self.reflection_repository.save_rule(rule)
                rules.append(rule)
        else:
            confidence = self.confidence_estimator.estimate(
                "Reflection suppressed due to low trigger score.",
                evidence_strength=0.0,
                retrieval_support=self._retrieval_support(context.retrieval_confidence),
                state_support=self._state_support(context),
            )

        after_confidence = confidence.overall_confidence if should_reflect else before_confidence
        utility_score = round(max(0.0, after_confidence - before_confidence), 6)
        return ReflectionResult(
            should_reflect=should_reflect,
            trigger_score=round(trigger_score, 6),
            signals=signals,
            reflection_memory=reflection_memory,
            rules=tuple(rules),
            confidence=confidence,
            state_updates={
                "last_reflection_trigger_score": round(trigger_score, 6),
                "last_failure_type": failure_type.value,
                "reflection_utility_score": utility_score,
            },
            trigger_reasons=trigger_reasons,
            before_confidence=before_confidence,
            after_confidence=after_confidence,
            utility_score=utility_score,
        )


    def trigger_reasons(
        self,
        context: ReflectionContext,
        signals: tuple[ReflectionSignal, ...],
        trigger_score: float,
        failure_type: FailureType,
    ) -> tuple[dict[str, object], ...]:
        """Return structured reasons explaining why reflection did or did not trigger."""
        reasons: list[dict[str, object]] = []
        affect_confidence = self._affect_support(context)
        retrieval_confidence = self._retrieval_support(context.retrieval_confidence)
        affect_uncertainty = 1.0 - affect_confidence
        low_affect_threshold = float(context.failure_metadata.get("affect_threshold", 0.30))
        low_confidence_threshold = float(context.failure_metadata.get("threshold", 0.30))
        audit_sample = bool(context.failure_metadata.get("audit_sample", False))
        reasons.append(
            {
                "reason": "affect_uncertainty",
                "confidence": round(affect_confidence, 6),
                "threshold": round(low_affect_threshold, 6),
                "triggered": affect_uncertainty >= (1.0 - low_affect_threshold),
            }
        )
        reasons.append(
            {
                "reason": "low_confidence",
                "confidence": round(retrieval_confidence, 6),
                "threshold": round(low_confidence_threshold, 6),
                "triggered": retrieval_confidence < low_confidence_threshold and (low_confidence_threshold <= 0.30 or audit_sample),
                "audit_sample": audit_sample,
            }
        )
        signal_types = {signal.signal_type for signal in signals}
        failure_reason = str(context.failure_metadata.get("reason", "")).lower()
        reasons.extend(
            (
                {
                    "reason": "tool_failure",
                    "confidence": round(trigger_score, 6),
                    "threshold": self.trigger_threshold,
                    "triggered": failure_type == FailureType.TOOL_FAILURE,
                },
                {
                    "reason": "contradiction",
                    "confidence": round(trigger_score, 6),
                    "threshold": self.trigger_threshold,
                    "triggered": (
                        failure_type in {FailureType.MEMORY_FAILURE, FailureType.STATE_CONFLICT, FailureType.EMOTIONAL_CONFLICT}
                        or "contradiction" in failure_reason
                        or "conflict" in failure_reason
                    ),
                },
                {
                    "reason": "hallucination",
                    "confidence": round(trigger_score, 6),
                    "threshold": self.trigger_threshold,
                    "triggered": failure_type == FailureType.HALLUCINATION_RISK,
                },
                {
                    "reason": "constraint_violation",
                    "confidence": round(trigger_score, 6),
                    "threshold": self.trigger_threshold,
                    "triggered": (
                        failure_type in {FailureType.PLANNING_FAILURE, FailureType.GOAL_CONFLICT}
                        or "constraint" in failure_reason
                        or "policy" in failure_reason
                    ),
                },
                {
                    "reason": "retrieval_ambiguity",
                    "confidence": round(trigger_score, 6),
                    "threshold": self.trigger_threshold,
                    "triggered": ReflectionSignalType.RETRIEVAL_AMBIGUITY in signal_types
                    and retrieval_confidence < low_confidence_threshold
                    and (low_confidence_threshold <= 0.30 or audit_sample),
                },
            )
        )
        return tuple(reasons)

    def _before_confidence(self, context: ReflectionContext) -> float:
        retrieval = self._retrieval_support(context.retrieval_confidence)
        state = self._state_support(context)
        return round(max(0.0, min(1.0, retrieval * 0.65 + state * 0.35)), 6)

    def compute_trigger_score(self, context: ReflectionContext, signals: tuple[ReflectionSignal, ...]) -> float:
        affect_uncertainty = 1.0 - self._affect_support(context)
        retrieval_uncertainty = 1.0 - self._retrieval_support(context.retrieval_confidence)
        contradiction_score = self._contradiction_score(context, signals, context.failure_type or FailureType.REASONING_FAILURE)
        score = (
            self.trigger_weights.affect_weight * affect_uncertainty
            + self.trigger_weights.retrieval_weight * retrieval_uncertainty
            + self.trigger_weights.contradiction_weight * contradiction_score
        )
        return max(0.0, min(1.0, round(score, 6)))

    def generate_reflection_text(self, context: ReflectionContext, failure_type: FailureType) -> str:
        reason = str(context.failure_metadata.get("reason", failure_type.value))
        meta_instruction = self.rule_extractor.extract_meta_instruction(reason)
        rejected = context.failure_metadata.get("rejected_action")
        rejected_part = f" Avoid repeating the rejected action: {rejected}." if rejected else ""
        return f"{meta_instruction} Failure type: {failure_type.value}. Reason: {reason}.{rejected_part}"

    def _build_signals(self, context: ReflectionContext, failure_type: FailureType) -> tuple[ReflectionSignal, ...]:
        signals = [
            ReflectionSignal.create(
                signal_type=FAILURE_TO_SIGNAL[failure_type],
                severity=float(context.failure_metadata.get("severity", 0.75)),
                confidence=0.85,
                source=ReflectionSource.FAILURE_CLASSIFIER.value,
                metadata={"failure_type": failure_type.value},
            )
        ]
        if context.retrieval_confidence and context.retrieval_confidence.confidence < 0.4:
            signals.append(
                ReflectionSignal.create(
                    ReflectionSignalType.RETRIEVAL_AMBIGUITY,
                    severity=1.0 - context.retrieval_confidence.confidence,
                    confidence=0.9,
                    source=ReflectionSource.RETRIEVAL_CONFIDENCE.value,
                )
            )
        for affect_signal in context.affect_signals:
            signal_type = getattr(affect_signal, "signal_type", "")
            if "dissonance" in str(signal_type) or "conflict" in str(signal_type):
                signals.append(
                    ReflectionSignal.create(
                        ReflectionSignalType.EMOTIONAL_DISSONANCE,
                        severity=float(getattr(affect_signal, "strength", 0.6)),
                        confidence=0.8,
                        source=ReflectionSource.AFFECT.value,
                        metadata={"affect_signal": str(signal_type)},
                    )
                )
        return tuple(signals)

    def _retrieval_support(self, retrieval_confidence: RetrievalConfidence | None) -> float:
        return retrieval_confidence.confidence if retrieval_confidence else 0.5

    def _affect_support(self, context: ReflectionContext) -> float:
        if context.affect_confidence is not None:
            return max(0.0, min(1.0, float(context.affect_confidence)))
        emotional_state = context.emotional_state or (context.cognitive_state.emotional_state if context.cognitive_state else None)
        if emotional_state is not None:
            return max(0.0, min(1.0, float(getattr(emotional_state, "confidence", 0.5))))
        return 0.5

    def _contradiction_score(
        self,
        context: ReflectionContext,
        signals: tuple[ReflectionSignal, ...],
        failure_type: FailureType,
    ) -> float:
        contradiction_types = {FailureType.MEMORY_FAILURE, FailureType.STATE_CONFLICT, FailureType.EMOTIONAL_CONFLICT}
        failure_reason = str(context.failure_metadata.get("reason", "")).lower()
        if failure_type in contradiction_types or "contradiction" in failure_reason or "conflict" in failure_reason:
            return max((signal.severity for signal in signals), default=float(context.failure_metadata.get("severity", 0.0)))
        return float(context.failure_metadata.get("contradiction_score", 0.0))

    def _evidence_strength(self, context: ReflectionContext) -> float:
        return min(1.0, len(context.retrieved_memories) / 3.0)

    def _state_support(self, context: ReflectionContext) -> float:
        snapshot = context.state_snapshot()
        if not snapshot:
            return 0.5
        confidence = snapshot.get("confidence_state", {})
        if isinstance(confidence, dict) and "confidence" in confidence:
            return max(0.0, min(1.0, float(confidence["confidence"])))
        return 0.65

    def _emotional_pressure(self, context: ReflectionContext) -> float:
        emotional_state = context.emotional_state or (context.cognitive_state.emotional_state if context.cognitive_state else None)
        if emotional_state is None:
            return 0.0
        return max(0.0, min(1.0, emotional_state.emotional_volatility))
