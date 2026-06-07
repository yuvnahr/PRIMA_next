"""Reflection signal generation."""

from __future__ import annotations

from affect.affect_types import ReflectionSignal
from affect.emotion_profile import EmotionProfile


def generate_reflection_signals(
    profile: EmotionProfile,
    dissonance_score: float,
    evolution: dict[str, float],
    volatility: float,
) -> tuple[ReflectionSignal, ...]:
    signals: list[ReflectionSignal] = []

    if dissonance_score >= 0.55:
        signals.append(
            ReflectionSignal(
                signal_type="emotional_dissonance",
                strength=dissonance_score,
                reason="Current affect diverges from recent emotional history.",
            )
        )

    if profile.confidence <= 0.25:
        signals.append(
            ReflectionSignal(
                signal_type="elevated_uncertainty",
                strength=round(1.0 - profile.confidence, 6),
                reason="Emotion profile is flat or weakly supported.",
            )
        )

    if profile.valence < -0.35 and evolution.get("acceleration", 0.0) > 0.25:
        signals.append(
            ReflectionSignal(
                signal_type="rapid_negative_drift",
                strength=min(1.0, abs(profile.valence) + evolution["acceleration"] / 2.0),
                reason="Negative affect is increasing quickly.",
            )
        )

    if volatility >= 0.55:
        signals.append(
            ReflectionSignal(
                signal_type="emotional_conflict",
                strength=volatility,
                reason="Recent emotional states are oscillating.",
            )
        )

    return tuple(signals)
