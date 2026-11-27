"""Lazy Hugging Face encoder for 28-label GoEmotions multilabel inference."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from affect.classifiers.calibration_artifact import encoder_identity, finite_parameter
from affect.classifiers.goemotions_adapter import GoEmotionsProfileAdapter
from affect.emotion_prediction import EmotionPrediction
from affect.emotion_profile import EmotionProfile
from affect.taxonomies.goemotions import LABELS
from config.frozen_json import read_frozen_json

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
    local_files_only: bool = False
    calibration_fingerprint: str | None = None
    thresholds_fingerprint: str | None = None
    expected_dev_hash: str | None = None
    expected_dev_ids_hash: str | None = None
    _calibration_provenance: dict[str, Any] = field(default_factory=dict, init=False)
    _thresholds_identity: str | None = field(default=None, init=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _device: str | None = field(default=None, init=False)
    _thresholds: dict[str, float] = field(default_factory=lambda: {label: 0.5 for label in LABELS}, init=False)
    _temperature: float = field(default=1.0, init=False)

    @classmethod
    def from_environment(cls, model_id: str) -> GoEmotionsEncoder:
        return cls(
            model_id=model_id,
            revision=os.getenv("PRIMA_AFFECT_MODEL_REVISION") or None,
            requested_device=os.getenv("PRIMA_AFFECT_DEVICE", "auto"),
            batch_size=int(os.getenv("PRIMA_AFFECT_BATCH_SIZE", "4")),
            thresholds_path=_optional_path("PRIMA_AFFECT_THRESHOLDS_PATH"),
            calibration_path=_optional_path("PRIMA_AFFECT_CALIBRATION_PATH"),
            calibration_fingerprint=os.getenv("PRIMA_AFFECT_CALIBRATION_FINGERPRINT"),
            thresholds_fingerprint=os.getenv("PRIMA_AFFECT_THRESHOLDS_FINGERPRINT"),
            local_files_only=os.getenv("PRIMA_AFFECT_LOCAL_FILES_ONLY", "false").lower() in {"1", "true", "yes"},
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
                predictions.append(EmotionPrediction.from_probabilities(probabilities, self._thresholds, self.model_id, self.revision, {"resolved_device": self._device, "batch_size": self.batch_size, "max_length": self.max_length, "calibration_temperature": self._temperature,
                    "calibration_provenance": dict(self._calibration_provenance),
                    "thresholds_fingerprint": self._thresholds_identity}))
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
            raise RuntimeError("GoEmotions encoder dependencies are listed in requirements.txt.") from exc
        device = self._resolve_device()
        revision = self.revision or PINNED_MODEL_REVISIONS.get(self.model_id)
        if revision is None:
            raise ValueError("Custom Hugging Face models require PRIMA_AFFECT_MODEL_REVISION.")
        if len(revision) < 7 or any(character not in "0123456789abcdefABCDEF" for character in revision):
            raise ValueError("Hugging Face model revisions must be immutable commit hashes.")
        kwargs = {"trust_remote_code": False, "local_files_only": self.local_files_only}
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=revision, **kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id, revision=revision, **kwargs).to(device)
        model.eval()
        labels = model.config.id2label
        ordered = tuple(labels.get(index, labels.get(str(index))) for index in range(len(LABELS)))
        if ordered != LABELS or len(labels) != len(LABELS):
            raise ValueError(f"Model label order does not match official GoEmotions order: {ordered!r}")
        resolved_revision = getattr(model.config, "_commit_hash", None) or revision
        if not isinstance(resolved_revision, str) or not resolved_revision.lower().startswith(revision.lower()):
            raise ValueError("Loaded model revision differs from requested encoder revision")
        revision = resolved_revision.lower()
        temperature, thresholds, calibration, threshold_id = self._validated_parameters(revision)
        self.revision, self._device = revision, device
        self._temperature, self._thresholds = temperature, thresholds
        self._calibration_provenance, self._thresholds_identity = calibration, threshold_id
        self._tokenizer, self._model = tokenizer, model

    def _artifact(self, path: Path, pin: str | None, revision: str, method: str) -> dict[str, Any]:
        if not pin:
            raise ValueError("Configured encoder artifact requires an independently pinned fingerprint")
        artifact = read_frozen_json(path, pin)
        expected = encoder_identity(
            self.model_id, revision, self.max_length,
            self.expected_dev_hash or artifact.get("dev_dataset_sha256", ""),
            self.expected_dev_ids_hash or artifact.get("dev_ids_sha256", ""),
        )
        if artifact.get("version") != 2 or artifact.get("method") != method or any(
            artifact.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("Encoder calibration artifact identity mismatch")
        return artifact

    def _validated_parameters(self, revision: str) -> tuple[float, dict[str, float], dict[str, Any], str | None]:
        temperature, thresholds = 1.0, {label: 0.5 for label in LABELS}
        provenance: dict[str, Any] = {}
        temperature_id = threshold_id = None
        if self.calibration_path is not None:
            artifact = self._artifact(self.calibration_path, self.calibration_fingerprint, revision, "scalar_temperature")
            temperature = finite_parameter(artifact.get("temperature"), lower=0.0, upper=float("inf"))
            if temperature == 0:
                raise ValueError("Calibration temperature must be positive")
            temperature_id = artifact["fingerprint"]
            provenance = {key: artifact[key] for key in ("model_id", "model_revision", "method", "split", "dev_dataset_sha256")}
            provenance["dev_ids_sha256"] = artifact["dev_ids_sha256"]
            provenance.update(calibrated=True, artifact_sha256=temperature_id, source="trained_probability")
        if self.thresholds_path is not None:
            artifact = self._artifact(self.thresholds_path, self.thresholds_fingerprint, revision, "per_label_f1")
            expected_transform = {"method": "sigmoid", "temperature_fingerprint": temperature_id}
            if artifact.get("probability_transform") != expected_transform:
                raise ValueError("Threshold probability transform identity mismatch")
            if provenance and any(artifact[key] != provenance[key] for key in ("dev_ids_sha256", "dev_dataset_sha256")):
                raise ValueError("Calibration and threshold development IDs differ")
            values = artifact.get("thresholds")
            if not isinstance(values, dict) or set(values) != set(LABELS):
                raise ValueError("Threshold artifact must contain every official GoEmotions label")
            thresholds = {label: finite_parameter(values[label], lower=0.0, upper=1.0) for label in LABELS}
            threshold_id = artifact["fingerprint"]
        return temperature, thresholds, provenance, threshold_id

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


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None
