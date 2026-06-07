"""Convert affect outputs into memory metadata without writing memory."""

from __future__ import annotations

from typing import Any

from affect.emotion_profile import EmotionProfile


class EmotionalMemoryAdapter:
    def to_metadata(
        self,
        profile: EmotionProfile,
        salience_score: float,
        dissonance_score: float,
    ) -> dict[str, Any]:
        intensity = max(profile.emotions.values()) if profile.emotions else 0.0
        return {
            "emotion": profile.dominant_emotion,
            "intensity": round(float(intensity), 6),
            "confidence": profile.confidence,
            "valence": profile.valence,
            "arousal": profile.arousal,
            "dominance": profile.dominance,
            "salience": salience_score,
            "dissonance": dissonance_score,
            "keywords": list(profile.emotional_keywords),
        }
