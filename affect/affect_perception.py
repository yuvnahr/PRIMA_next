"""PRIMA's deterministic GoEmotions probability-to-label decision controller."""

from __future__ import annotations

from collections.abc import Mapping

from affect.emotion_prediction import EmotionPrediction
from affect.taxonomies.goemotions import LABELS


class GoEmotionsDecisionController:
    """Apply calibrated probabilities, per-label thresholds, and neutral gating."""

    def __init__(self, thresholds: Mapping[str, float] | None = None) -> None:
        self.thresholds = {label: float((thresholds or {}).get(label, 0.5)) for label in LABELS}
        if any(not 0 < value < 1 for value in self.thresholds.values()):
            raise ValueError("GoEmotions thresholds must be strictly between zero and one.")

    def decide(self, probabilities: Mapping[str, float], *, model_id: str, model_revision: str | None = None, metadata: Mapping[str, object] | None = None) -> EmotionPrediction:
        final = tuple(label for label in LABELS if float(probabilities[label]) >= self.thresholds[label]) or ("neutral",)
        return EmotionPrediction.from_probabilities(
            probabilities,
            self.thresholds,
            model_id,
            model_revision,
            {**dict(metadata or {}), "prima_controller": "goemotions-v2", "final_prima_labels": list(final), "change_reason": "official_multilabel_threshold"},
            selected_labels=final,
        )
