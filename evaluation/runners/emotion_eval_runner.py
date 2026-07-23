"""Supervised emotion evaluation runner for PRIMA affect."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from affect import DynamicAffectEngine
from evaluation.metrics.affect_metrics import classification_report

DEFAULT_GOLD_PATH = Path("evaluation/datasets/emotion_gold.json")
DEFAULT_OUTPUT_PATH = Path("evaluation/results/emotion_eval_results.json")
CORE_LABELS = ("joy", "sadness", "fear", "anger", "surprise", "disgust", "trust", "anticipation")
LABEL_ALIASES = {
    "sad": "sadness",
    "joy_ecstasy": "joy",
    "amazement_surprise": "surprise",
    "interest_vigilance": "surprise",
    "disgust_loathing": "disgust",
    "senerity": "trust",
    "serenity": "trust",
    "admire": "trust",
    "acceptance": "trust",
    "positive": "joy",
    "negative": "sadness",
    "love": "joy",
    "optimism": "anticipation",
    "gratitude": "trust",
    "approval": "trust",
    "caring": "trust",
    "excitement": "joy",
    "grief": "sadness",
    "nervousness": "fear",
    "annoyance": "anger",
    "disapproval": "disgust",
    "realization": "surprise",
    "curiosity": "surprise",
}


class EmotionModel(Protocol):
    """Callable emotion model used by the evaluation runner."""

    name: str

    def predict(self, text: str) -> str:
        """Return one canonical emotion label."""


@dataclass(slots=True)
class PrimaAffectModel:
    """PRIMA affect adapter for supervised evaluation."""

    engine: DynamicAffectEngine = field(default_factory=DynamicAffectEngine)
    name: str = "prima_affect"

    def predict(self, text: str) -> str:
        """Classify text with the current PRIMA affect engine."""
        return canonical_label(self.engine.get_emotion_profile(text).dominant_emotion)


@dataclass(frozen=True, slots=True)
class KeywordBaseline:
    """Deterministic local baseline used when external model weights are unavailable."""

    name: str
    lexicon: dict[str, tuple[str, ...]]

    def predict(self, text: str) -> str:
        """Predict the label with the highest lexical evidence."""
        lowered = text.lower()
        scores: dict[str, int] = {label: 0 for label in CORE_LABELS}
        for label, terms in self.lexicon.items():
            canonical = canonical_label(label)
            for term in terms:
                if term in lowered:
                    scores[canonical] = scores.get(canonical, 0) + max(1, len(term.split()))
        best_label = max(scores, key=lambda label: scores[label])
        return best_label if scores[best_label] > 0 else "joy"


@dataclass(slots=True)
class OptionalTransformersBaseline:
    """Optional local-files-only transformers baseline with deterministic fallback."""

    name: str
    model_name: str
    label_mapper: Callable[[str], str]
    fallback: KeywordBaseline
    _pipeline: Any | None = None
    available: bool = False

    def __post_init__(self) -> None:
        if os.getenv("PRIMA_ENABLE_TRANSFORMER_BASELINES", "0") != "1":
            self._pipeline = None
            self.available = False
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                top_k=1,
                local_files_only=True,
            )
            self.available = True
        except Exception:
            self._pipeline = None
            self.available = False

    def predict(self, text: str) -> str:
        """Predict using local transformer weights when available, otherwise fallback."""
        if self._pipeline is None:
            return self.fallback.predict(text)
        result = self._pipeline(text)
        first = result[0][0] if result and isinstance(result[0], list) else result[0]
        return canonical_label(self.label_mapper(str(first.get("label", ""))))

    def metadata(self) -> dict[str, Any]:
        """Return baseline availability metadata."""
        return {
            "model_name": self.model_name,
            "local_transformer_available": self.available,
            "fallback": self.fallback.name if not self.available else None,
        }


class EmotionEvalRunner:
    """Run supervised emotion evaluation against PRIMA and baselines."""

    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_GOLD_PATH,
        output_path: str | Path = DEFAULT_OUTPUT_PATH,
        models: tuple[EmotionModel, ...] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.models = models or default_models()

    def run(self) -> dict[str, Any]:
        """Evaluate all configured models and write metrics to disk."""
        samples = self._load_samples()
        expected = [canonical_label(str(sample["emotion"])) for sample in samples]
        texts = [str(sample["text"]) for sample in samples]
        results: dict[str, Any] = {
            "dataset": str(self.dataset_path),
            "sample_count": len(samples),
            "labels": list(CORE_LABELS),
            "models": {},
            "success_criteria": {"accuracy": 0.75, "macro_f1": 0.70},
        }

        for model in self.models:
            predictions = [canonical_label(model.predict(text)) for text in texts]
            metrics = classification_report(expected, predictions, labels=list(CORE_LABELS))
            payload = {
                **metrics,
                "meets_success_criteria": metrics["accuracy"] >= 0.75 and metrics["macro_f1"] >= 0.70,
            }
            metadata = getattr(model, "metadata", None)
            if callable(metadata):
                payload["metadata"] = metadata()
            results["models"][model.name] = payload

        primary = results["models"].get("prima_affect", {})
        results["primary_model"] = "prima_affect"
        results["accuracy"] = primary.get("accuracy", 0.0)
        results["macro_f1"] = primary.get("macro_f1", 0.0)
        results["meets_success_criteria"] = primary.get("meets_success_criteria", False)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results

    def _load_samples(self) -> list[dict[str, str]]:
        raw = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Emotion gold dataset must be a list of objects.")
        samples: list[dict[str, str]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict) or "text" not in item or "emotion" not in item:
                raise ValueError(f"Invalid emotion gold sample at index {index}.")
            samples.append({"text": str(item["text"]), "emotion": canonical_label(str(item["emotion"]))})
        return samples


def canonical_label(label: str) -> str:
    """Map legacy and external labels onto the core evaluation label set."""
    normalized = label.strip().lower().replace(" ", "_")
    canonical = LABEL_ALIASES.get(normalized, normalized)
    if canonical not in CORE_LABELS:
        raise ValueError(f"Unsupported emotion label: {label!r}")
    return canonical


def default_models() -> tuple[EmotionModel, ...]:
    """Return PRIMA plus DistilBERT and GoEmotions baseline adapters."""
    distilbert_fallback = KeywordBaseline("distilbert_emotion_keyword_baseline", _distilbert_emotion_lexicon())
    goemotions_fallback = KeywordBaseline("goemotions_keyword_baseline", _goemotions_lexicon())
    return (
        PrimaAffectModel(),
        OptionalTransformersBaseline(
            name="distilbert_emotion",
            model_name="bhadresh-savani/distilbert-base-uncased-emotion",
            label_mapper=canonical_label,
            fallback=distilbert_fallback,
        ),
        OptionalTransformersBaseline(
            name="goemotions_baseline",
            model_name="SamLowe/roberta-base-go_emotions",
            label_mapper=canonical_label,
            fallback=goemotions_fallback,
        ),
    )


def _distilbert_emotion_lexicon() -> dict[str, tuple[str, ...]]:
    return {
        "joy": ("joy", "happy", "pleasure", "delighted", "promoted", "celebrate", "thrilled", "proud"),
        "sadness": ("sad", "grief", "heartbroken", "died", "loss", "crying", "lonely", "devastated"),
        "fear": ("afraid", "scared", "nervous", "worried", "terrified", "panic", "unsafe", "dread"),
        "anger": ("angry", "mad", "furious", "outraged", "unfair", "yelled", "betrayed", "livid"),
        "surprise": ("surprise", "wonder", "amazed", "astonished", "unexpected", "startled", "shocked"),
        "disgust": ("disgust", "mold", "gross", "filthy", "nauseated", "repulsed", "rotten"),
        "trust": ("trust", "safe", "reliable", "support", "honest", "loyal", "dependable", "secure"),
        "anticipation": ("expect", "upcoming", "hopeful", "looking forward", "counting down", "awaiting", "eager"),
    }


def _goemotions_lexicon() -> dict[str, tuple[str, ...]]:
    return {
        "joy": ("joy", "amusement", "approval", "excitement", "gratitude", "happy", "proud", "wonderful"),
        "sadness": ("sadness", "grief", "remorse", "disappointment", "heartbroken", "mourning", "tears"),
        "fear": ("fear", "nervousness", "anxious", "panic", "danger", "unsafe", "frightened"),
        "anger": ("anger", "annoyance", "disapproval", "furious", "irritated", "unfair", "betrayed"),
        "surprise": ("surprise", "realization", "curiosity", "wonder", "amazed", "astonished", "unexpected"),
        "disgust": ("disgust", "gross", "filthy", "repulsed", "rotten", "nauseated", "vile"),
        "trust": ("caring", "approval", "trust", "safe", "supported", "reassured", "dependable"),
        "anticipation": ("optimism", "desire", "eager", "hopeful", "expect", "upcoming", "awaiting"),
    }


def run_emotion_eval() -> dict[str, Any]:
    """Run the default supervised emotion evaluation."""
    return EmotionEvalRunner().run()


if __name__ == "__main__":
    print(json.dumps(run_emotion_eval(), indent=2))
