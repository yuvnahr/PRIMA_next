"""Failure source classification."""

from __future__ import annotations

from reflection.reflection_types import FailureType


class FailureClassifier:
    def classify(self, failure_reason: str, metadata: dict[str, object] | None = None) -> FailureType:
        text = f"{failure_reason} {metadata or {}}".lower()
        if any(token in text for token in ("tool", "api", "timeout", "invalid action", "syntax", "nothing happens")):
            return FailureType.TOOL_FAILURE
        if any(token in text for token in ("memory", "context missing", "stale memory")):
            return FailureType.MEMORY_FAILURE
        if any(token in text for token in ("retrieval", "ambiguous", "no relevant", "low confidence")):
            return FailureType.RETRIEVAL_FAILURE
        if any(token in text for token in ("plan", "planning", "wrong step")):
            return FailureType.PLANNING_FAILURE
        if any(token in text for token in ("hallucination", "unsupported", "not grounded", "invented")):
            return FailureType.HALLUCINATION_RISK
        if any(token in text for token in ("goal", "objective", "intent")):
            return FailureType.GOAL_CONFLICT
        if any(token in text for token in ("state", "inconsistent", "conflict")):
            return FailureType.STATE_CONFLICT
        if any(token in text for token in ("emotion", "dissonance", "volatility", "affect")):
            return FailureType.EMOTIONAL_CONFLICT
        return FailureType.REASONING_FAILURE
