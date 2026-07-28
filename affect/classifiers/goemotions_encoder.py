"""Lazy Hugging Face encoder for 28-label GoEmotions multilabel inference."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from affect.classifiers.goemotions_adapter import GoEmotionsProfileAdapter
from affect.emotion_prediction import EmotionPrediction
from affect.emotion_profile import EmotionProfile
from affect.taxonomies.goemotions import LABELS

PINNED_MODEL_REVISIONS = {
    "SamLowe/roberta-base-go_emotions": "d750483",
    "microsoft/deberta-v3-small": "a59be8aa63396e73dbb45a1487e4cde4be98bfa4",
}


@dataclass(slots=True)
class GoEmotionsEncoder:
    """A lazy, batched sigmoid classifier; optional packages load only on first inference."""

    model_id: str
    revision: str | None = None
    requested_device: str = "auto"
    batch_size: int = 4
    max_length: int = 128
    thresholds_path: Path | None = None
    calibration_path: Path | None = None
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _device: str | None = field(default=None, init=False)
    _thresholds: dict[str, float] = field(default_factory=lambda: {label: 0.5 for label in LABELS}, init=False)
    _temperature: float = field(default=1.0, init=False)

    @classmethod
    def from_environment(cls, model_id: str) -> "GoEmotionsEncoder":
        return cls(
            model_id=model_id,
            revision=os.getenv("PRIMA_AFFECT_MODEL_REVISION") or None,
            requested_device=os.getenv("PRIMA_AFFECT_DEVICE", "auto"),
            batch_size=int(os.getenv("PRIMA_AFFECT_BATCH_SIZE", "4")),
            thresholds_path=_optional_path("PRIMA_AFFECT_THRESHOLDS_PATH"),
            calibration_path=_optional_path("PRIMA_AFFECT_CALIBRATION_PATH"),
        )

    def predict(self, text: str) -> EmotionPrediction:
        return self.predict_many([text])[0]

    def classify(self, text: str) -> EmotionProfile:
        return GoEmotionsProfileAdapter(self).classify(text)

    def predict_many(self, texts: list[str]) -> list[EmotionPrediction]:
        if not texts:
            return []
        self._load()
        torch = self._torch()
        predictions: list[EmotionPrediction] = []
        for start in range(0, len(texts), self.batch_size):
            encoded = self._tokenizer(texts[start : start + self.batch_size], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                logits = self._model(**encoded).logits / self._temperature
                rows = torch.sigmoid(logits).detach().cpu().tolist()
            for row in rows:
                probabilities = {self._label_for_index(index): float(value) for index, value in enumerate(row)}
                predictions.append(EmotionPrediction.from_probabilities(probabilities, self._thresholds, self.model_id, self.revision, {"resolved_device": self._device, "batch_size": self.batch_size, "max_length": self.max_length, "calibration_temperature": self._temperature}))
        return predictions

    def close(self) -> None:
        """Release model references predictably for long-lived benchmark processes."""
        self._model = self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("GoEmotions encoder requires requirements-goemotions.txt.") from exc
        self._device = self._resolve_device()
        revision = self.revision or PINNED_MODEL_REVISIONS.get(self.model_id)
        if revision is None:
            raise ValueError("Custom Hugging Face models require PRIMA_AFFECT_MODEL_REVISION.")
        if len(revision) < 7 or any(character not in "0123456789abcdefABCDEF" for character in revision):
            raise ValueError("Hugging Face model revisions must be immutable commit hashes.")
        self.revision = revision
        kwargs = {"trust_remote_code": False, "local_files_only": os.getenv("PRIMA_AFFECT_LOCAL_FILES_ONLY", "false").lower() in {"1", "true", "yes"}}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=revision, **kwargs)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id, revision=revision, **kwargs).to(self._device)
        self._model.eval()
        self._validate_label_order()
        self._thresholds = _load_thresholds(self.thresholds_path)
        self._temperature = _load_temperature(self.calibration_path)

    def _torch(self) -> Any:
        import torch

        return torch

    def _resolve_device(self) -> str:
        torch = self._torch()
        requested = self.requested_device.lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("PRIMA_AFFECT_DEVICE requests CUDA but CUDA is unavailable.")
        if requested not in {"cpu", "cuda"} and not requested.startswith("cuda:"):
            raise ValueError(f"Unsupported affect device: {self.requested_device}")
        return self.requested_device

    def _label_for_index(self, index: int) -> str:
        value = self._model.config.id2label.get(index, self._model.config.id2label.get(str(index)))
        if value not in LABELS:
            raise ValueError(f"Model label order is not compatible with official GoEmotions: index {index} is {value!r}.")
        return str(value)

    def _validate_label_order(self) -> None:
        ordered = tuple(self._label_for_index(index) for index in range(len(LABELS)))
        if ordered != LABELS:
            raise ValueError(f"Model label order does not match official GoEmotions order: {ordered!r}")


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None


def _load_thresholds(path: Path | None) -> dict[str, float]:
    if path is None:
        return {label: 0.5 for label in LABELS}
    if not path.is_file():
        raise FileNotFoundError(f"Configured threshold artifact not found: {path}")
    values = json.loads(path.read_text(encoding="utf-8")).get("thresholds", {})
    if set(values) != set(LABELS):
        raise ValueError("Threshold artifact must contain every official GoEmotions label.")
    return {label: float(values[label]) for label in LABELS}


def _load_temperature(path: Path | None) -> float:
    if path is None:
        return 1.0
    if not path.is_file():
        raise FileNotFoundError(f"Configured calibration artifact not found: {path}")
    temperature = float(json.loads(path.read_text(encoding="utf-8")).get("temperature", 1.0))
    if temperature <= 0:
        raise ValueError("Calibration temperature must be positive.")
    return temperature
