"""Temporal affect evolution calculations."""

from __future__ import annotations

from affect.emotion_history import EmotionHistory
from affect.emotion_profile import EmotionProfile
from affect.pad_model import PADState


def update_momentum(
    current: dict[str, float],
    profile: EmotionProfile,
    decay: float = 0.82,
) -> dict[str, float]:
    decayed = {emotion: round(score * decay, 6) for emotion, score in current.items() if score * decay > 0.01}
    dominant = profile.dominant_emotion
    decayed[dominant] = round(decayed.get(dominant, 0.0) + profile.confidence, 6)
    return decayed


def compute_volatility(history: EmotionHistory, window: int = 6) -> float:
    recent = history.get_recent(window)
    if len(recent) < 2:
        return 0.0
    distances = [
        recent[index].pad_state.distance(recent[index - 1].pad_state)
        for index in range(1, len(recent))
    ]
    return round(min(1.0, sum(distances) / len(distances)), 6)


def compute_stability(volatility: float, confidence: float) -> float:
    return round(max(0.0, min(1.0, (1.0 - volatility) * (0.5 + confidence / 2.0))), 6)


def detect_dissonance(history: EmotionHistory, current_pad: PADState, confidence: float) -> float:
    recent = history.get_recent(5)
    if len(recent) < 3:
        return 0.0
    historical_pad = history.average_pad(recent)
    distance = historical_pad.distance(current_pad)
    valence_flip = historical_pad.pleasure * current_pad.pleasure < -0.05
    score = min(1.0, distance / 2.0)
    if valence_flip:
        score += 0.25
    if confidence < 0.35:
        score += 0.15
    return round(min(1.0, score), 6)


def evolution_metrics(history: EmotionHistory, baseline: PADState) -> dict[str, float]:
    return {
        "velocity": history.compute_velocity(),
        "acceleration": history.compute_acceleration(),
        "drift": history.compute_drift(baseline),
    }
