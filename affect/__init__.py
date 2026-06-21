"""PRIMA-NEXT affect subsystem."""

from typing import Any

__all__ = [
    "AffectUpdate",
    "DynamicAffectEngine",
    "EmotionProfile",
    "EmotionalState",
    "PADState",
    "get_emotion_profile",
]


def __getattr__(name: str) -> Any:
    if name in {"DynamicAffectEngine", "get_emotion_profile"}:
        from affect.affect_engine import DynamicAffectEngine, get_emotion_profile

        return {"DynamicAffectEngine": DynamicAffectEngine, "get_emotion_profile": get_emotion_profile}[name]
    if name == "AffectUpdate":
        from affect.affect_types import AffectUpdate

        return AffectUpdate
    if name == "EmotionProfile":
        from affect.emotion_profile import EmotionProfile

        return EmotionProfile
    if name == "PADState":
        from affect.pad_model import PADState

        return PADState
    if name == "EmotionalState":
        from state.emotional_state import EmotionalState

        return EmotionalState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
