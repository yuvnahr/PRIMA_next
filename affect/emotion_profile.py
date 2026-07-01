"""Emotion profile domain model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


def _immutable_scores(scores: Mapping[str, float]) -> Mapping[str, float]:
    cleaned = {
        str(key): round(float(value), 6)
        for key, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if float(value) > 0
    }
    return MappingProxyType(cleaned)


@dataclass(frozen=True, slots=True)
class EmotionProfile:
    """Analysis of a single input utterance."""

    emotions: Mapping[str, float]
    dominant_emotion: str
    confidence: float
    emotional_keywords: tuple[str, ...]
    valence: float
    arousal: float
    dominance: float

    def __post_init__(self) -> None:
        immutable_scores = _immutable_scores(self.emotions)
        object.__setattr__(self, "emotions", immutable_scores)
        object.__setattr__(self, "emotional_keywords", tuple(self.emotional_keywords))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "valence", max(-1.0, min(1.0, float(self.valence))))
        object.__setattr__(self, "arousal", max(-1.0, min(1.0, float(self.arousal))))
        object.__setattr__(self, "dominance", max(-1.0, min(1.0, float(self.dominance))))

    @classmethod
    def from_scores(
        cls,
        scores: Mapping[str, float],
        emotional_keywords: tuple[str, ...] = (),
        confidence: float | None = None,
    ) -> EmotionProfile:
        if scores:
            def _key_fn(k: str) -> float:
                return float(scores.get(k, 0.0))

            dominant = max(scores.keys(), key=_key_fn)
        else:
            dominant = "neutral"
        computed_confidence = confidence if confidence is not None else profile_confidence(scores)
        valence, arousal, dominance = scores_to_pad(scores)
        return cls(
            emotions=scores,
            dominant_emotion=dominant,
            confidence=computed_confidence,
            emotional_keywords=emotional_keywords,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "emotions": dict(self.emotions),
            "dominant_emotion": self.dominant_emotion,
            "confidence": self.confidence,
            "emotional_keywords": list(self.emotional_keywords),
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }


EMOTION_PAD_MAP: dict[str, tuple[float, float, float]] = {
    "joy": (0.9, 0.65, 0.55),
    "joy_ecstasy": (1.0, 0.85, 0.65),
    "trust": (0.65, 0.25, 0.5),
    "admire": (0.75, 0.45, 0.45),
    "acceptance": (0.35, -0.15, 0.2),
    "anticipation": (0.25, 0.55, 0.15),
    "interest_vigilance": (0.2, 0.75, 0.35),
    "surprise": (0.05, 0.85, -0.1),
    "amazement_surprise": (0.35, 0.9, 0.1),
    "senerity": (0.65, -0.45, 0.45),
    "sadness": (-0.75, -0.35, -0.55),
    "sad": (-0.75, -0.35, -0.55),
    "fear": (-0.75, 0.85, -0.75),
    "anger": (-0.65, 0.8, 0.55),
    "disgust": (-0.65, 0.45, 0.15),
    "disgust_loathing": (-0.85, 0.55, 0.1),
    "boredom": (-0.35, -0.75, -0.3),
    "distraction": (-0.15, 0.25, -0.35),
    "positive": (0.65, 0.25, 0.35),
    "negative": (-0.65, 0.35, -0.35),
    "uncertainty": (-0.25, 0.5, -0.45),
    "litigious": (-0.45, 0.45, 0.35),
    "model_strong": (0.35, 0.15, 0.45),
    "model_weak": (-0.25, 0.15, -0.45),
}


def scores_to_pad(scores: Mapping[str, float]) -> tuple[float, float, float]:
    total = sum(value for value in scores.values() if value > 0)
    if total <= 0:
        return (0.0, 0.0, 0.0)

    pleasure = 0.0
    arousal = 0.0
    dominance = 0.0
    for emotion, score in scores.items():
        p, a, d = EMOTION_PAD_MAP.get(emotion, (0.0, 0.0, 0.0))
        weight = max(0.0, score) / total
        pleasure += p * weight
        arousal += a * weight
        dominance += d * weight

    return (round(pleasure, 6), round(arousal, 6), round(dominance, 6))


def profile_confidence(scores: Mapping[str, float]) -> float:
    """Entropy-based confidence: flat profiles are low confidence."""
    import math

    positive = [float(score) for score in scores.values() if score > 0]
    total = sum(positive)
    if total <= 0 or len(positive) <= 1:
        return 1.0 if total > 0 else 0.0
    probs = [score / total for score in positive]
    entropy = -sum(prob * math.log(prob + 1e-12) for prob in probs)
    max_entropy = math.log(len(probs))
    return round(max(0.0, min(1.0, 1.0 - entropy / max_entropy)), 6)
