"""Interchangeable GoEmotions inference systems."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Protocol

from affect.affect_engine import DynamicAffectEngine
from affect.affect_perception import GoEmotionsDecisionController
from affect.classifiers.goemotions_adapter import GoEmotionsProfileAdapter
from affect.emotion_prediction import EmotionPrediction
from affect.taxonomies.goemotions import LABELS
from benchmarks.goemotions.prompts import build_prompt
from benchmarks.goemotions.schemas import GOEMOTIONS_RESPONSE_SCHEMA
from llm.llm_client import LLMClient


class GoEmotionsSystem(Protocol):
    name: str

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        """Return raw output plus serializable system metadata."""


@dataclass(frozen=True, slots=True)
class QwenZeroShotSystem:
    provider: str = "ollama"
    model: str = "qwen3.5:4b"
    name: str = "qwen_zero_shot"

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        response = LLMClient(provider_name=self.provider).chat(build_prompt(text, labels), model=self.model, temperature=0, max_tokens=128)
        return response.text, {"provider_response": response.raw, "schema": None}


@dataclass(frozen=True, slots=True)
class QwenSchemaSystem(QwenZeroShotSystem):
    name: str = "qwen_schema"

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        response = LLMClient(provider_name=self.provider).chat(build_prompt(text, labels, structured=True), model=self.model, temperature=0, max_tokens=128, response_schema=GOEMOTIONS_RESPONSE_SCHEMA)
        return response.text, {"provider_response": response.raw, "schema": GOEMOTIONS_RESPONSE_SCHEMA}


@dataclass(slots=True)
class PrimaQwenSystem:
    """Run Qwen labels through PRIMA's stateful affect engine without altering benchmark labels."""

    provider: str = "ollama"
    model: str = "qwen3.5:4b"
    name: str = "prima_qwen"
    _current: "_CurrentPrediction" = field(init=False, repr=False)
    _engine: DynamicAffectEngine = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._current = _CurrentPrediction()
        self._engine = DynamicAffectEngine(classifier=GoEmotionsProfileAdapter(self._current))

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        response = LLMClient(provider_name=self.provider).chat(build_prompt(text, labels), model=self.model, temperature=0, max_tokens=128)
        raw, metadata = response.text, {"provider_response": response.raw, "schema": None}
        self._current.value = _prediction_from_qwen(raw, labels, self.model)
        update = self._engine.process(text)
        return raw, {**metadata, "prima_update": update.to_dict(), "prima_prediction": self._current.value.to_dict()}


@dataclass(slots=True)
class _CurrentPrediction:
    value: EmotionPrediction | None = None

    def predict(self, text: str) -> EmotionPrediction:
        if self.value is None:
            raise RuntimeError("No Qwen prediction is available for PRIMA affect processing.")
        return self.value


def _prediction_from_qwen(raw: str, labels: list[str], model: str) -> EmotionPrediction:
    try:
        payload = json.loads(raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        selected = payload.get("labels", []) if isinstance(payload, dict) else []
        if not isinstance(selected, list) or not all(isinstance(label, str) and label in labels for label in selected):
            selected = []
        if "neutral" in selected and len(selected) > 1:
            selected = []
    except json.JSONDecodeError:
        selected = []
    probabilities = {label: 1.0 if label in selected else 0.0 for label in LABELS}
    return EmotionPrediction.from_probabilities(probabilities, {}, model, metadata={"source": "qwen_label_set", "probabilities_calibrated": False})



@dataclass(slots=True)
class EncoderSystem:
    """Official-label encoder system; batching is delegated to the encoder."""

    name: str
    model: str
    device: str = "auto"
    batch_size: int = 4
    thresholds_path: Any = None
    calibration_path: Any = None

    def __post_init__(self) -> None:
        from affect.classifiers.goemotions_encoder import GoEmotionsEncoder

        self._encoder = GoEmotionsEncoder(self.model, requested_device=self.device, batch_size=self.batch_size, thresholds_path=self.thresholds_path, calibration_path=self.calibration_path)

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        return self.predict_many([text], labels)[0]

    def predict_many(self, texts: list[str], labels: list[str]) -> list[tuple[str, dict[str, Any]]]:
        predictions = self._encoder.predict_many(texts)
        return [(json.dumps({"labels": list(prediction.selected_labels)}), {"prediction": prediction.to_dict()}) for prediction in predictions]


class HybridSystem(EncoderSystem):
    """Encoder first, with explicit schema-constrained Qwen adjudication when uncertain."""

    def __init__(self, model: str, provider: str, device: str, batch_size: int, thresholds_path: Any = None, calibration_path: Any = None) -> None:
        super().__init__("goemotions_hybrid", model, device, batch_size, thresholds_path, calibration_path)
        self._adjudicator = QwenSchemaSystem(provider, os.getenv("PRIMA_AFFECT_LLM_MODEL", "qwen3.5:4b"))
        self._uncertainty_threshold = float(os.getenv("PRIMA_AFFECT_UNCERTAINTY_THRESHOLD", "0.55"))

    def predict_many(self, texts: list[str], labels: list[str]) -> list[tuple[str, dict[str, Any]]]:
        predictions = self._encoder.predict_many(texts)
        results: list[tuple[str, dict[str, Any]]] = []
        for text, prediction in zip(texts, predictions, strict=True):
            metadata: dict[str, Any] = {"prediction": prediction.to_dict(), "adjudicated": False}
            if prediction.uncertainty >= self._uncertainty_threshold:
                raw, adjudication = self._adjudicator.predict(text, labels)
                metadata.update({"adjudicated": True, "adjudication": adjudication})
                results.append((raw, metadata))
            else:
                results.append((json.dumps({"labels": list(prediction.selected_labels)}), metadata))
        return results


@dataclass(slots=True)
class PrimaGoEmotionsSystem:
    """Encoder probabilities become PRIMA-controller labels before affect-state updates."""

    model: str
    device: str = "auto"
    batch_size: int = 4
    thresholds_path: Any = None
    calibration_path: Any = None
    name: str = "prima_goemotions"
    _encoder: Any = field(init=False, repr=False)
    _current: "_CurrentPrediction" = field(init=False, repr=False)
    _engine: DynamicAffectEngine = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from affect.classifiers.goemotions_encoder import GoEmotionsEncoder

        self._encoder = GoEmotionsEncoder(self.model, requested_device=self.device, batch_size=self.batch_size, thresholds_path=self.thresholds_path, calibration_path=self.calibration_path)
        self._current = _CurrentPrediction()
        self._engine = DynamicAffectEngine(classifier=GoEmotionsProfileAdapter(self._current))

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        return self.predict_many([text], labels)[0]

    def predict_many(self, texts: list[str], labels: list[str]) -> list[tuple[str, dict[str, Any]]]:
        encoder_predictions = self._encoder.predict_many(texts)
        results: list[tuple[str, dict[str, Any]]] = []
        for text, encoder_prediction in zip(texts, encoder_predictions, strict=True):
            controller = GoEmotionsDecisionController(encoder_prediction.thresholds)
            final_prediction = controller.decide(encoder_prediction.probabilities, model_id=encoder_prediction.model_id, model_revision=encoder_prediction.model_revision, metadata={"encoder_prediction": encoder_prediction.to_dict()})
            self._current.value = final_prediction
            update = self._engine.process(text)
            results.append((json.dumps({"labels": list(final_prediction.selected_labels)}), {"prediction": final_prediction.to_dict(), "prima_update": update.to_dict(), "prima_controller_changed_labels": list(encoder_prediction.selected_labels) != list(final_prediction.selected_labels)}))
        return results
