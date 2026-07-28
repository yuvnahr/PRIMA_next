"""Interchangeable GoEmotions inference systems."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Protocol

from affect.affect_engine import DynamicAffectEngine
from affect.affect_perception import GoEmotionsDecisionController
from affect.classifiers.goemotions_adapter import GoEmotionsProfileAdapter
from affect.emotion_classifier import LegacyAffectClassifier
from affect.emotion_prediction import EmotionPrediction
from affect.taxonomies.goemotions import LABELS
from benchmarks.goemotions.prompts import build_prompt
from benchmarks.goemotions.schemas import GOEMOTIONS_RESPONSE_SCHEMA, parse_label_response
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
        prompt = build_prompt(text, labels)
        response = LLMClient(provider_name=self.provider).chat(prompt, model=self.model, temperature=0, max_tokens=128)
        return response.text, {"provider_response": response.raw, "schema": None, "benchmark_prompt": prompt}


@dataclass(frozen=True, slots=True)
class QwenDefinitionsSystem(QwenZeroShotSystem):
    name: str = "qwen_definitions"

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        prompt = build_prompt(text, labels, include_definitions=True)
        response = LLMClient(provider_name=self.provider).chat(prompt, model=self.model, temperature=0, max_tokens=128)
        return response.text, {"provider_response": response.raw, "schema": None, "benchmark_prompt": prompt}


@dataclass(frozen=True, slots=True)
class QwenSchemaSystem(QwenZeroShotSystem):
    name: str = "qwen_schema"

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        prompt = build_prompt(text, labels, include_definitions=True, structured=True)
        response = LLMClient(provider_name=self.provider).chat(prompt, model=self.model, temperature=0, max_tokens=128, response_schema=GOEMOTIONS_RESPONSE_SCHEMA)
        return response.text, {"provider_response": response.raw, "schema": GOEMOTIONS_RESPONSE_SCHEMA, "benchmark_prompt": prompt}


@dataclass(frozen=True, slots=True)
class QwenWithPrimaTelemetrySystem(QwenSchemaSystem):
    """Preserve Qwen labels while recording an independent PRIMA state update."""

    name: str = "qwen_with_prima_telemetry"

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        raw, metadata = QwenSchemaSystem.predict(self, text, labels)
        normalized, parse_error = parse_label_response(raw, labels)
        prediction = _prediction_from_labels(normalized or frozenset({"neutral"}), self.model)
        current = _CurrentPrediction(prediction)
        update = DynamicAffectEngine(classifier=GoEmotionsProfileAdapter(current)).process(text)
        return raw, {
            **metadata,
            "raw_qwen_labels": _raw_labels(raw),
            "normalized_labels": sorted(normalized),
            "prima_derived_labels": sorted(normalized),
            "final_scored_labels": sorted(normalized),
            "qwen_parse_error": parse_error,
            "prima_update": update.to_dict(),
            "prima_decision": {"added_labels": [], "removed_labels": [], "reason": "telemetry_only_preserve_qwen", "preserved_baseline": True},
        }


@dataclass(frozen=True, slots=True)
class PrimaQwenSystem(QwenSchemaSystem):
    """Augment neutral Qwen predictions with independent PRIMA lexical evidence."""

    name: str = "prima_qwen"

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        raw, metadata = QwenSchemaSystem.predict(self, text, labels)
        normalized, parse_error = parse_label_response(raw, labels)
        update = DynamicAffectEngine(classifier=LegacyAffectClassifier()).process(text)
        candidate = update.profile.dominant_emotion
        derived = {candidate} if update.profile.emotional_keywords and candidate in {"anger", "disgust", "fear", "joy", "sadness", "surprise"} else set()
        baseline = set(normalized)
        final = set(baseline)
        reason = "preserve_qwen_labels"
        if (not baseline or baseline == {"neutral"}) and derived:
            final.update(derived)
            reason = "add_prima_lexical_emotion"
        elif not baseline:
            final.add("neutral")
            reason = "recover_parse_failure_as_neutral"
        added, removed = sorted(final - baseline), sorted(baseline - final)
        final_raw = json.dumps({"labels": sorted(final)})
        return final_raw, {
            **metadata,
            "raw_qwen_response": raw,
            "raw_qwen_labels": _raw_labels(raw),
            "normalized_labels": sorted(normalized),
            "prima_derived_labels": sorted(derived),
            "final_scored_labels": sorted(final),
            "qwen_parse_error": parse_error,
            "prima_update": update.to_dict(),
            "prima_decision": {"added_labels": added, "removed_labels": removed, "reason": reason, "preserved_baseline": not added and not removed},
        }


@dataclass(slots=True)
class _CurrentPrediction:
    value: EmotionPrediction | None = None

    def predict(self, text: str) -> EmotionPrediction:
        if self.value is None:
            raise RuntimeError("No Qwen prediction is available for PRIMA affect processing.")
        return self.value


def _raw_labels(raw: str) -> list[str]:
    try:
        payload = json.loads(raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        values = payload.get("labels", []) if isinstance(payload, dict) else []
        return [value for value in values if isinstance(value, str)] if isinstance(values, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def _prediction_from_labels(selected: frozenset[str], model: str) -> EmotionPrediction:
    probabilities = {label: 1.0 if label in selected else 0.0 for label in LABELS}
    return EmotionPrediction.from_probabilities(probabilities, {}, model, metadata={"source": "qwen_label_set", "probabilities_calibrated": False}, selected_labels=tuple(selected))



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

    def __post_init__(self) -> None:
        from affect.classifiers.goemotions_encoder import GoEmotionsEncoder

        self._encoder = GoEmotionsEncoder(self.model, requested_device=self.device, batch_size=self.batch_size, thresholds_path=self.thresholds_path, calibration_path=self.calibration_path)

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        return self.predict_many([text], labels)[0]

    def predict_many(self, texts: list[str], labels: list[str]) -> list[tuple[str, dict[str, Any]]]:
        encoder_predictions = self._encoder.predict_many(texts)
        results: list[tuple[str, dict[str, Any]]] = []
        for text, encoder_prediction in zip(texts, encoder_predictions, strict=True):
            controller = GoEmotionsDecisionController(encoder_prediction.thresholds)
            final_prediction = controller.decide(encoder_prediction.probabilities, model_id=encoder_prediction.model_id, model_revision=encoder_prediction.model_revision, metadata={"encoder_prediction": encoder_prediction.to_dict()})
            current = _CurrentPrediction(final_prediction)
            update = DynamicAffectEngine(classifier=GoEmotionsProfileAdapter(current)).process(text)
            results.append((json.dumps({"labels": list(final_prediction.selected_labels)}), {"prediction": final_prediction.to_dict(), "prima_update": update.to_dict(), "prima_controller_changed_labels": list(encoder_prediction.selected_labels) != list(final_prediction.selected_labels)}))
        return results
