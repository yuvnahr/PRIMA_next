"""Configuration-driven, lazy affect classifier selection."""

from __future__ import annotations

import os

from affect.classifiers.goemotions_adapter import GoEmotionsProfileAdapter
from affect.emotion_classifier import LegacyAffectClassifier
from affect.interfaces import EmotionClassifier


def create_affect_classifier() -> EmotionClassifier:
    """Resolve the requested backend without importing or downloading models by default."""

    backend = os.getenv("PRIMA_AFFECT_BACKEND", "legacy").strip().lower()
    if backend == "legacy":
        return LegacyAffectClassifier()
    if backend not in {"goemotions_pretrained", "goemotions_deberta", "goemotions_hybrid"}:
        raise ValueError(f"Unsupported PRIMA_AFFECT_BACKEND: {backend}")
    try:
        from affect.classifiers.goemotions_encoder import GoEmotionsEncoder

        model_id = os.getenv("PRIMA_AFFECT_MODEL_ID") or ("SamLowe/roberta-base-go_emotions" if backend == "goemotions_pretrained" else "microsoft/deberta-v3-small")
        return GoEmotionsProfileAdapter(GoEmotionsEncoder.from_environment(model_id=model_id))
    except Exception:
        if os.getenv("PRIMA_AFFECT_ALLOW_FALLBACK", "false").lower() in {"1", "true", "yes"}:
            return LegacyAffectClassifier()
        raise


def resolved_backend() -> str:
    """Return the configured backend name without creating or downloading a model."""

    return os.getenv("PRIMA_AFFECT_BACKEND", "legacy").strip().lower()
