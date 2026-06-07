"""Reflection confidence model and estimator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReflectionConfidence:
    reflection_quality: float
    reflection_relevance: float
    evidence_strength: float
    retrieval_support: float
    state_support: float
    overall_confidence: float


class ReflectionConfidenceEstimator:
    def estimate(
        self,
        reflection_text: str,
        evidence_strength: float,
        retrieval_support: float,
        state_support: float,
    ) -> ReflectionConfidence:
        words = [word for word in reflection_text.split() if word.strip()]
        quality = min(1.0, len(words) / 18.0)
        relevance = 1.0 if any(token in reflection_text.lower() for token in ("must", "avoid", "verify", "search", "state")) else 0.55
        overall = (
            quality * 0.25
            + relevance * 0.25
            + evidence_strength * 0.20
            + retrieval_support * 0.15
            + state_support * 0.15
        )
        return ReflectionConfidence(
            reflection_quality=round(max(0.0, min(1.0, quality)), 6),
            reflection_relevance=round(max(0.0, min(1.0, relevance)), 6),
            evidence_strength=round(max(0.0, min(1.0, evidence_strength)), 6),
            retrieval_support=round(max(0.0, min(1.0, retrieval_support)), 6),
            state_support=round(max(0.0, min(1.0, state_support)), 6),
            overall_confidence=round(max(0.0, min(1.0, overall)), 6),
        )
