"""Truthfully scoped GoEmotions systems executed through the public runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from affect.affect_engine import DynamicAffectEngine
from affect.classifiers.goemotions_adapter import GoEmotionsProfileAdapter
from affect.emotion_classifier import LegacyAffectClassifier
from affect.emotion_prediction import EmotionPrediction
from affect.taxonomies.goemotions import LABELS
from benchmarks.goemotions.prompts import build_prompt
from benchmarks.goemotions.schemas import GOEMOTIONS_RESPONSE_SCHEMA, parse_label_response
from llm.generation_config import GenerationConfig
from llm.llm_client import LLMClient
from runtime.contracts import ExecutionOutcome, ExecutionProfile, PrimaRequest, RuntimeComponent, TaskKind
from runtime.prima_runtime import PrimaRuntime

LABEL_POLICY_VERSION = "preserve_labels_v1"
LABEL_AUDIT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class ClassifierSettings:
    """Request-scoped classifier settings; no environment lookup is permitted."""

    device: str = "auto"
    batch_size: int = 4
    thresholds_path: Path | None = None
    calibration_path: Path | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("classifier batch_size must be positive")


class GoEmotionsSystem(Protocol):
    name: str
    prompt_variant: str
    system_family: str

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        """Return raw output plus serializable system metadata."""

    def use_client(self, client: LLMClient) -> Any:
        """Reuse one campaign-owned inference client."""


class _PredictionSource(Protocol):
    last_raw: str
    last_metadata: dict[str, Any]

    def predict(self, text: str) -> EmotionPrediction: ...


@dataclass(slots=True)
class _LLMPredictionSource:
    generation: GenerationConfig
    include_definitions: bool
    schema_constrained: bool
    client: LLMClient = field(repr=False)
    prompt_variant: str = field(default="definitions_schema", kw_only=True)
    last_raw: str = field(default="", init=False)
    last_metadata: dict[str, Any] = field(default_factory=dict, init=False)

    def predict(self, text: str) -> EmotionPrediction:
        prompt = build_prompt(
            text, list(LABELS), include_definitions=self.include_definitions, structured=self.schema_constrained,
            variant=self.prompt_variant,
        )
        if hasattr(self.client.provider, "reset_request_metrics"):
            self.client.provider.reset_request_metrics()
        response = self.client.chat(
            prompt,
            generation_config=self.generation,
            response_schema=GOEMOTIONS_RESPONSE_SCHEMA if self.schema_constrained else None,
        )
        selected, parse_error = parse_label_response(response.text)
        self.last_raw = response.text
        self.last_metadata = {
            "benchmark_prompt": prompt,
            "prompt_variant": self.prompt_variant,
            "provider_response": response.raw,
            "provider_usage": dict(response.usage or {}),
            "probabilities_available": False,
            "confidence_available": False,
            "confidence": None,
            "confidence_reason": "label_only_or_unavailable_scores",
            "logical_classification_provider_calls": 1,
            "classification_provider": (
                self.client.provider.request_metrics() if hasattr(self.client.provider, "request_metrics")
                else {"available": False, "reason": "provider_request_instrumentation_unavailable"}
            ),
            "baseline_labels": sorted(selected),
            "baseline_parse_error": parse_error,
            "baseline_empty_output": not response.text.strip(),
            "raw_model_response": response.text,
            "final_scored_labels": sorted(selected),
            "label_audit_schema_version": LABEL_AUDIT_SCHEMA_VERSION,
            "label_policy_version": LABEL_POLICY_VERSION,
            "parse_recovery": {
                "applied": parse_error is not None,
                "labels": ["neutral"] if parse_error else [],
                "scope": "runtime_only_not_scored",
            },
            "label_correction": {
                "policy_version": LABEL_POLICY_VERSION,
                "evidence_source": "unavailable",
                "accepted": False,
                "added_labels": [],
                "removed_labels": [],
                "reason": "baseline_parse_failure" if parse_error else "no_validated_correction_policy",
                "preserved_baseline": True,
            },
        }
        return _prediction_from_labels(selected or frozenset({"neutral"}), self.generation.model)


@dataclass(slots=True)
class _StaticPredictionSource:
    value: EmotionPrediction
    raw: str
    metadata: dict[str, Any] = field(default_factory=dict)
    last_raw: str = field(default="", init=False)
    last_metadata: dict[str, Any] = field(default_factory=dict, init=False)

    def predict(self, text: str) -> EmotionPrediction:
        self.last_raw = self.raw
        self.last_metadata = dict(self.metadata)
        return self.value


@dataclass(slots=True)
class _BoundedLLMPredictionSource(_LLMPredictionSource):
    """Audit the disabled legacy heuristic without changing the model label set."""

    def predict(self, text: str) -> EmotionPrediction:
        baseline_prediction = _LLMPredictionSource.predict(self, text)
        profile = LegacyAffectClassifier().classify(text)
        candidate = profile.dominant_emotion
        derived = (
            {candidate}
            if profile.emotional_keywords and candidate in {"anger", "disgust", "fear", "joy", "sadness", "surprise"}
            else set()
        )
        decision = {
            **self.last_metadata["label_correction"],
            "evidence_source": profile.evidence_source,
            "reason": "unvalidated_legacy_correction_suppressed",
        }
        self.last_metadata.update(
            {
                "affect_derived_labels": sorted(derived),
                "legacy_policy_version": "bounded_lexical_v0",
                "legacy_policy_enabled": False,
                "label_correction": decision,
                "affect_decision": dict(decision),
                "prima_decision": dict(decision),
            }
        )
        return baseline_prediction


@dataclass(slots=True)
class _RuntimeClassificationSystem:
    generation: GenerationConfig
    name: str
    system_family: str
    prompt_variant: str = field(default="definitions_schema", kw_only=True)
    _client_instance: LLMClient | None = field(default=None, init=False, repr=False)
    _runtime_local: threading.local = field(default_factory=threading.local, init=False, repr=False)

    def use_client(self, client: LLMClient) -> _RuntimeClassificationSystem:
        """Reuse one campaign-owned inference client."""

        self._client_instance = client
        return self

    def _client(self) -> LLMClient:
        if self._client_instance is None:
            from benchmarks.campaign.provider_session import _BoundedProvider

            self._client_instance = LLMClient(provider_name=self.generation.provider)
            self._client_instance.provider = _BoundedProvider(self._client_instance.provider, 1)
        return self._client_instance

    def provider_request_metrics(self) -> dict[str, Any]:
        provider = self._client().provider
        return provider.request_metrics() if hasattr(provider, "request_metrics") else {"available": False}

    def _execute(self, text: str, source: _PredictionSource) -> tuple[str, dict[str, Any]]:
        adapter = getattr(self._runtime_local, "adapter", None)
        runtime = getattr(self._runtime_local, "runtime", None)
        if adapter is None or runtime is None:
            adapter = GoEmotionsProfileAdapter(source)
            runtime = PrimaRuntime(
                affect_engine=DynamicAffectEngine(classifier=adapter),
                llm_client=self._client(),
                generation_config=self.generation,
                maintenance_enabled=False,
            )
            self._runtime_local.adapter = adapter
            self._runtime_local.runtime = runtime
        else:
            adapter.predictor = source
            adapter.last_prediction = None
        response = asyncio.run(
            runtime.execute(
                PrimaRequest(
                    task_kind=TaskKind.EMOTION_CLASSIFICATION,
                    profile=ExecutionProfile.AFFECT_ONLY,
                    input_text=text,
                    generation_config=self.generation,
                    metadata={"benchmark": "goemotions", "system": self.name},
                )
            )
        )
        prediction = adapter.last_prediction
        if response.outcome is not ExecutionOutcome.CLASSIFIED or prediction is None:
            raise RuntimeError(f"Canonical emotion-classification route failed: {response.errors}")
        forbidden = {
            RuntimeComponent.QUERY_REWRITER,
            RuntimeComponent.DENSE_RETRIEVAL,
            RuntimeComponent.SPARSE_RETRIEVAL,
            RuntimeComponent.TEMPORAL_RETRIEVAL,
            RuntimeComponent.GRAPH_TRAVERSAL,
            RuntimeComponent.GRAPH_REASONING,
            RuntimeComponent.FUSION,
            RuntimeComponent.RERANKER,
            RuntimeComponent.PLANNER,
            RuntimeComponent.WORLD_MODEL,
            RuntimeComponent.TOOL_EXECUTOR,
            RuntimeComponent.MODEL_EXECUTOR,
            RuntimeComponent.REFLECTION,
        }
        if forbidden & set(response.diagnostics.executed_components):
            raise RuntimeError("GoEmotions activated a component outside the bounded affect route.")
        return source.last_raw, {
            "confidence_available": False,
            "confidence": None,
            "confidence_reason": "probability_eligibility_requires_benchmark_split_identity",
            **source.last_metadata,
            "prediction": {**prediction.to_dict(), "confidence": None, "confidence_available": False},
            "final_scored_labels": source.last_metadata.get("final_scored_labels", list(prediction.selected_labels)),
            "system_name": self.name,
            "system_family": self.system_family,
            "runtime": {
                "task_kind": response.task_kind.value,
                "profile": response.profile.value,
                "route_name": response.diagnostics.route_name,
                "outcome": response.outcome.value,
                "executed_components": [item.value for item in response.diagnostics.executed_components],
                "latency_ms": response.diagnostics.latency_ms,
                "model_usage": dict(response.diagnostics.model_usage),
            },
            "scope": "affect/classification component benchmark; not retrieval, QA, tools, or full architecture",
            "raw_model_response_hash": hashlib.sha256(
                str(source.last_metadata.get("raw_model_response", source.last_raw)).encode("utf-8")
            ).hexdigest(),
        }


@dataclass(slots=True)
class ModelOnlyZeroShotSystem(_RuntimeClassificationSystem):
    name: str = "model_only_zero_shot"
    system_family: str = "model_only"

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        raw, metadata = self._execute(text, _LLMPredictionSource(self.generation, False, False, self._client(), prompt_variant=self.prompt_variant))
        metadata["final_scored_labels"] = list(metadata.get("baseline_labels", []))
        return raw, metadata


@dataclass(slots=True)
class SchemaConstrainedModelOnlySystem(_RuntimeClassificationSystem):
    name: str = "schema_constrained_model_only"
    system_family: str = "model_only"

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        raw, metadata = self._execute(text, _LLMPredictionSource(self.generation, True, True, self._client(), prompt_variant=self.prompt_variant))
        metadata["final_scored_labels"] = list(metadata.get("baseline_labels", []))
        return raw, metadata


@dataclass(slots=True)
class AffectTelemetrySystem(SchemaConstrainedModelOnlySystem):
    """Run affect telemetry while preserving the model's parsed label set."""

    name: str = "affect_telemetry_preserve_labels"
    system_family: str = "affect_telemetry"

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        raw, metadata = SchemaConstrainedModelOnlySystem.predict(self, text, labels)
        baseline = metadata.get("baseline_labels", [])
        decision = {**metadata["label_correction"], "reason": "telemetry_only_preserve_model_labels"}
        metadata.update(
            {
                "final_scored_labels": list(baseline),
                "affect_decision": decision,
                "prima_decision": {**decision, "reason": "telemetry_only_preserve_qwen"},
            }
        )
        return raw, metadata


@dataclass(slots=True)
class BoundedAffectDecisionSystem(SchemaConstrainedModelOnlySystem):
    """Deprecated legacy entry point; v1 preserves labels and disables lexical edits."""

    name: str = "bounded_prima_affect_decision"
    system_family: str = "bounded_affect_decision"

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        return self._execute(text, _BoundedLLMPredictionSource(self.generation, True, True, self._client(), prompt_variant=self.prompt_variant))


@dataclass(slots=True)
class EncoderSystem(_RuntimeClassificationSystem):
    """A separately attributed trained encoder baseline routed through classification."""

    classifier: ClassifierSettings = field(default_factory=ClassifierSettings)
    _encoder: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from affect.classifiers.goemotions_encoder import GoEmotionsEncoder

        self._encoder = GoEmotionsEncoder(
            self.generation.model,
            revision=self.generation.revision,
            requested_device=self.classifier.device,
            batch_size=self.classifier.batch_size,
            thresholds_path=self.classifier.thresholds_path,
            calibration_path=self.classifier.calibration_path,
        )

    def predict(self, text: str, labels: list[str]) -> tuple[str, dict[str, Any]]:
        return self.predict_many([text], labels)[0]

    def predict_many(self, texts: list[str], labels: list[str]) -> list[tuple[str, dict[str, Any]]]:
        results = []
        for text, prediction in zip(texts, self._encoder.predict_many(texts), strict=True):
            raw = json.dumps({"labels": list(prediction.selected_labels)})
            results.append(
                self._execute(
                    text,
                    _StaticPredictionSource(
                        prediction,
                        raw,
                        {
                            "trained_baseline": "encoder",
                            "system_family": "trained_baseline",
                            "wrapper_attribution": False,
                            "probabilities_available": True,
                            "evidence_source": "trained_probability",
                            "probability_provenance": self._calibration_provenance(prediction),
                            "logical_classification_provider_calls": 0,
                            "classification_provider": {
                                "available": True, "attempts": 0, "successes": 0, "retries": 0,
                                "cache_hits": 0, "actual_tokens": 0, "actual_tokens_available": True,
                                "inference_started": None, "inference_finished": None,
                            },
                        },
                    ),
                )
            )
        return results

    def _calibration_provenance(self, prediction: EmotionPrediction) -> dict[str, Any]:
        path = self.classifier.calibration_path
        if path is None:
            return {"source": "trained_probability", "calibrated": False}
        content = path.read_bytes()
        calibration = json.loads(content)
        return {
            "source": "trained_probability",
            "calibrated": (calibration.get("model_id") == prediction.model_id
                           and calibration.get("model_revision") == prediction.model_revision
                           and prediction.model_revision is not None
                           and calibration.get("temperature") == (prediction.metadata or {}).get("calibration_temperature")),
            "model_id": calibration.get("model_id"), "model_revision": calibration.get("model_revision"),
            "method": calibration.get("method"), "split": "dev",
            "dev_dataset_sha256": calibration.get("dev_dataset_sha256"),
            "artifact_sha256": hashlib.sha256(content).hexdigest(),
        }


def _legacy_generation(provider: str, model: str, *, schema: bool) -> GenerationConfig:
    from llm.generation_config import StructuredOutputMode

    return GenerationConfig(
        model=model,
        provider=provider,
        structured_output=StructuredOutputMode.JSON_SCHEMA if schema else StructuredOutputMode.NONE,
    )


class QwenZeroShotSystem(ModelOnlyZeroShotSystem):
    """Compatibility constructor for the truthfully named model-only system."""

    def __init__(self, provider: str = "ollama", model: str = "qwen3.5:4b") -> None:
        super().__init__(_legacy_generation(provider, model, schema=False))


class QwenDefinitionsSystem(SchemaConstrainedModelOnlySystem):
    def __init__(self, provider: str = "ollama", model: str = "qwen3.5:4b") -> None:
        super().__init__(_legacy_generation(provider, model, schema=True))


QwenSchemaSystem = QwenDefinitionsSystem


class QwenWithPrimaTelemetrySystem(AffectTelemetrySystem):
    def __init__(self, provider: str = "ollama", model: str = "qwen3.5:4b") -> None:
        super().__init__(_legacy_generation(provider, model, schema=True))


class PrimaQwenSystem(BoundedAffectDecisionSystem):
    def __init__(self, provider: str = "ollama", model: str = "qwen3.5:4b") -> None:
        super().__init__(_legacy_generation(provider, model, schema=True))


PrimaGoEmotionsSystem = EncoderSystem


def _prediction_from_labels(selected: frozenset[str], model: str) -> EmotionPrediction:
    probabilities = {label: 1.0 if label in selected else 0.0 for label in LABELS}
    return EmotionPrediction.from_probabilities(
        probabilities,
        {},
        model,
        metadata={"source": "model_label_set", "probabilities_calibrated": False},
        selected_labels=tuple(selected),
    )
