"""Bridge complete GoEmotions predictions into the legacy PRIMA profile."""

from __future__ import annotations

from affect.emotion_prediction import EmotionPrediction
from affect.emotion_profile import EmotionProfile
from affect.taxonomies.goemotions import PRIMA_CORE_MAP


class GoEmotionsProfileAdapter:
    """Compatibility adapter that keeps fine labels in ``predict`` and core PAD in ``classify``."""

    def __init__(self, predictor: object) -> None:
        self.predictor = predictor
        self.last_prediction: EmotionPrediction | None = None

    def predict(self, text: str) -> EmotionPrediction:
        prediction = self.predictor.predict(text)  # type: ignore[attr-defined]
        if not isinstance(prediction, EmotionPrediction):
            raise TypeError("GoEmotions predictor must return EmotionPrediction.")
        return prediction

    def classify(self, text: str) -> EmotionProfile:
        prediction = self.predict(text)
        self.last_prediction = prediction
        scores: dict[str, float] = {}
        for label, probability in prediction.probabilities.items():
            core = PRIMA_CORE_MAP.get(label)
            if core is None:
                raise ValueError(f"No PRIMA-core mapping for GoEmotions label {label!r}.")
            scores[core] = scores.get(core, 0.0) + probability
        if scores.get("neutral", 0.0) >= max((value for label, value in scores.items() if label != "neutral"), default=0.0):
            return EmotionProfile.from_scores({}, confidence=prediction.confidence)
        scores.pop("neutral", None)
        total = sum(scores.values())
        return EmotionProfile.from_scores({label: value / total for label, value in scores.items()} if total else {}, confidence=prediction.confidence)
