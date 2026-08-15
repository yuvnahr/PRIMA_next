"""Configuration-driven, lazy affect classifier selection."""

from __future__ import annotations

import os
from pathlib import Path

from affect.classifiers.goemotions_adapter import GoEmotionsProfileAdapter
from affect.emotion_classifier import LegacyAffectClassifier
from affect.interfaces import EmotionClassifier


def create_affect_classifier(
    backend: str | None = None,
    model_id: str | None = None,
    allow_fallback: bool | None = None,
    *,
    revision: str | None = None,
    device: str = "auto",
    batch_size: int = 4,
    thresholds_path: str | None = None,
    calibration_path: str | None = None,
    local_files_only: bool = False,
) -> EmotionClassifier:
    """Resolve the requested backend without importing or downloading models by default."""

    backend = (backend or os.getenv("PRIMA_AFFECT_BACKEND") or "legacy").strip().lower()
    if backend == "legacy":
        return LegacyAffectClassifier()
    if backend not in {"goemotions_pretrained", "goemotions_deberta", "goemotions_hybrid"}:
        raise ValueError(f"Unsupported PRIMA_AFFECT_BACKEND: {backend}")
    try:
        from affect.classifiers.goemotions_encoder import GoEmotionsEncoder

        model_id = model_id or os.getenv("PRIMA_AFFECT_MODEL_ID") or ("SamLowe/roberta-base-go_emotions" if backend == "goemotions_pretrained" else "microsoft/deberta-v3-small")
        return GoEmotionsProfileAdapter(GoEmotionsEncoder(
            model_id=model_id,
            revision=revision,
            requested_device=device,
            batch_size=batch_size,
            thresholds_path=Path(thresholds_path) if thresholds_path else None,
            calibration_path=Path(calibration_path) if calibration_path else None,
            local_files_only=local_files_only,
        ))
    except Exception:
        fallback = (
            os.getenv("PRIMA_AFFECT_ALLOW_FALLBACK", "false").lower() in {"1", "true", "yes"}
            if allow_fallback is None else allow_fallback
        )
        if fallback:
            return LegacyAffectClassifier()
        raise


def resolved_backend() -> str:
    """Return the configured backend name without creating or downloading a model."""

    return os.getenv("PRIMA_AFFECT_BACKEND", "legacy").strip().lower()
