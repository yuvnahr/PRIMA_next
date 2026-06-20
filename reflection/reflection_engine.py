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
        trigger_threshold: float = 0.45,
    ) -> None:
        self.reflection_repository = reflection_repository or ReflectionRepository()
        self.failure_classifier = failure_classifier or FailureClassifier()
        self.rule_extractor = rule_extractor or RuleExtractor()
        self.confidence_estimator = confidence_estimator or ReflectionConfidenceEstimator()
        self.trigger_threshold = trigger_threshold

    def evaluate(self, context: ReflectionContext) -> ReflectionResult:
        failure_type = context.failure_type or self.failure_classifier.classify(
            str(context.failure_metadata.get("reason", "")),
            context.failure_metadata,
        )
        signals = self._build_signals(context, failure_type)
        trigger_score = self.compute_trigger_score(context, signals)
        should_reflect = trigger_score >= self.trigger_threshold

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

        return ReflectionResult(
            should_reflect=should_reflect,
            trigger_score=round(trigger_score, 6),
            signals=signals,
            reflection_memory=reflection_memory,
            rules=tuple(rules),
            confidence=confidence,
            state_updates={"last_reflection_trigger_score": round(trigger_score, 6), "last_failure_type": failure_type.value},
        )

    def compute_trigger_score(self, context: ReflectionContext, signals: tuple[ReflectionSignal, ...]) -> float:
        failure_severity = max((signal.severity for signal in signals), default=0.0)
        retrieval_pressure = 1.0 - self._retrieval_support(context.retrieval_confidence)
        emotional_pressure = self._emotional_pressure(context)
        state_pressure = 1.0 - self._state_support(context)
        history_pressure = min(1.0, len(context.reflection_history) / 5.0)
        score = (
            failure_severity * 0.35
            + retrieval_pressure * 0.25
            + emotional_pressure * 0.15
            + state_pressure * 0.15
            + history_pressure * 0.10
        )
        return max(0.0, min(1.0, score))

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
