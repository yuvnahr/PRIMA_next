"""Model-independent immutable probabilistic emotion prediction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from affect.taxonomies.goemotions import LABELS


@dataclass(frozen=True, slots=True)
class EmotionPrediction:
    probabilities: Mapping[str, float]
    selected_labels: tuple[str, ...]
    dominant_label: str
    confidence: float
    uncertainty: float
    entropy: float
    margin: float
    thresholds: Mapping[str, float]
    model_id: str
    model_revision: str | None = None
    taxonomy: str = "goemotions"
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.taxonomy != "goemotions" or set(self.probabilities) != set(LABELS):
            raise ValueError("GoEmotions predictions require exactly the official 28-label probability vector.")
        probabilities = {label: float(self.probabilities[label]) for label in LABELS}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
            raise ValueError("Emotion probabilities must be finite values in [0, 1].")
        selected = tuple(label for label in LABELS if label in self.selected_labels)
        if set(selected) != set(self.selected_labels):
            raise ValueError("Selected labels must be official GoEmotions labels.")
        if self.dominant_label not in LABELS:
            raise ValueError("dominant_label must be an official GoEmotions label.")
        object.__setattr__(self, "probabilities", MappingProxyType(probabilities))
        object.__setattr__(self, "thresholds", MappingProxyType({label: float(self.thresholds.get(label, 0.5)) for label in LABELS}))
        object.__setattr__(self, "selected_labels", selected)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))

    @classmethod
    def from_probabilities(cls, probabilities: Mapping[str, float], thresholds: Mapping[str, float], model_id: str, model_revision: str | None = None, metadata: Mapping[str, object] | None = None, selected_labels: tuple[str, ...] | None = None) -> "EmotionPrediction":
        ordered = {label: float(probabilities[label]) for label in LABELS}
        selected = selected_labels if selected_labels is not None else tuple(label for label in LABELS if ordered[label] >= float(thresholds.get(label, 0.5)))
        selected = selected or ("neutral",)
        ranked = sorted(ordered.values(), reverse=True)
        entropy = -sum(p * math.log(p + 1e-12) + (1 - p) * math.log(1 - p + 1e-12) for p in ordered.values()) / len(ordered)
        return cls(ordered, selected, max(LABELS, key=ordered.__getitem__), max(ordered.values()), min(1.0, entropy / math.log(2)), entropy, ranked[0] - ranked[1], thresholds, model_id, model_revision, metadata=metadata)

    def to_dict(self) -> dict[str, object]:
        return {"probabilities": dict(self.probabilities), "selected_labels": list(self.selected_labels), "dominant_label": self.dominant_label, "confidence": self.confidence, "uncertainty": self.uncertainty, "entropy": self.entropy, "margin": self.margin, "thresholds": dict(self.thresholds), "model_id": self.model_id, "model_revision": self.model_revision, "taxonomy": self.taxonomy, "metadata": dict(self.metadata or {})}
