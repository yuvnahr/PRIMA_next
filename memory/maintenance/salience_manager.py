"""Memory salience scoring."""

from __future__ import annotations


class SalienceManager:
    def score(
        self,
        emotion_intensity: float = 0.0,
        retrieval_frequency: float = 0.0,
        novelty: float = 0.0,
        state_relevance: float = 0.0,
    ) -> float:
        value = (
            emotion_intensity * 0.35
            + retrieval_frequency * 0.25
            + novelty * 0.20
            + state_relevance * 0.20
        )
        return round(max(0.0, min(1.0, value)), 6)
