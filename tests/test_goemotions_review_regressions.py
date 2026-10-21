"""B1 regressions: preserve labels, response identity, parse failures and route scope."""

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from affect.emotion_classifier import LegacyAffectClassifier
from affect.emotion_profile import EmotionProfile
from affect.taxonomies.goemotions import LABELS
from benchmarks.goemotions import experiment
from benchmarks.goemotions.dataset import GoEmotionsExample
from benchmarks.goemotions.systems import (
    LABEL_POLICY_VERSION,
    AffectTelemetrySystem,
    BoundedAffectDecisionSystem,
    ClassifierSettings,
    PrimaQwenSystem,
    QwenDefinitionsSystem,
    QwenSchemaSystem,
    QwenWithPrimaTelemetrySystem,
    QwenZeroShotSystem,
)
from llm.generation_config import GenerationConfig
from llm.llm_client import LLMClient
from llm.llm_types import LLMResponse

CASES = json.loads((Path(__file__).parent / "fixtures/goemotions_review/preserve_labels.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("system_type", [AffectTelemetrySystem, BoundedAffectDecisionSystem])
def test_preserve_labels_and_raw_response_audit(monkeypatch, case, system_type) -> None:
    monkeypatch.setattr(
        "llm.llm_client.LLMClient.chat",
        lambda self, prompt, **kwargs: LLMResponse(text=case["raw"], raw={"fixture": True}),
    )
    system = system_type(GenerationConfig(model="fixture", provider="ollama"))
    raw, metadata = system.predict(case["text"], list(LABELS))
    assert raw == metadata["raw_model_response"] == case["raw"]
    assert metadata["raw_model_response_hash"] == hashlib.sha256(raw.encode()).hexdigest()
    assert metadata["baseline_labels"] == metadata["final_scored_labels"] == case["expected"]
    assert metadata["label_policy_version"] == LABEL_POLICY_VERSION
    decision = metadata["label_correction"]
    assert decision["added_labels"] == decision["removed_labels"] == []
    assert not decision["accepted"] and decision["preserved_baseline"]
    if system_type is BoundedAffectDecisionSystem:
        assert metadata["legacy_policy_version"] == "bounded_lexical_v0"
        assert not metadata["legacy_policy_enabled"]
        assert decision["evidence_source"] == case["evidence_source"]
    else:
        assert decision["evidence_source"] == "unavailable"
    assert metadata["affect_decision"]["policy_version"] == LABEL_POLICY_VERSION
    assert metadata["prima_decision"]["policy_version"] == LABEL_POLICY_VERSION
    record = experiment._classify_response(
        GoEmotionsExample(example_id=case["id"], text=case["text"], labels=frozenset({"neutral"})),
        list(LABELS), raw, metadata, "fixture"
    )
    assert record["id"] == case["id"]
    assert record["response_hash"] == metadata["raw_model_response_hash"]
    assert record["predicted_labels"] == case["expected"]
    failed_parse = not case["expected"]
    assert bool(record["parse_error"]) == failed_parse
    assert metadata["parse_recovery"]["applied"] == failed_parse
    assert record["same_response_comparison_eligible"] is not failed_parse
    if failed_parse:
        assert metadata["prediction"]["selected_labels"] == ["neutral"]
    assert not (
        {"query_rewriter", "dense_retrieval", "sparse_retrieval", "temporal_retrieval", "graph_traversal",
         "graph_reasoning", "fusion", "reranker", "planner", "world_model", "tool_executor", "model_executor", "reflection"}
        & set(metadata["runtime"]["executed_components"])
    )


@pytest.mark.parametrize(
    "constructor",
    [QwenZeroShotSystem, QwenDefinitionsSystem, QwenSchemaSystem, QwenWithPrimaTelemetrySystem, PrimaQwenSystem],
)
def test_compatibility_constructors_preserve_mixed_labels(monkeypatch, constructor) -> None:
    raw = '{"labels":["neutral","joy"]}'
    monkeypatch.setattr(
        "llm.llm_client.LLMClient.chat", lambda self, prompt, **kwargs: LLMResponse(text=raw, raw={})
    )
    returned, metadata = constructor().predict("I am angry.", list(LABELS))
    assert returned == raw
    assert metadata["baseline_labels"] == metadata["final_scored_labels"] == ["joy", "neutral"]


def test_classifier_provenance_does_not_change_legacy_scores() -> None:
    classifier = LegacyAffectClassifier()
    for text, source in (("I am happy", "literal_span"), ("The book is on the table.", "hash_neighbour")):
        profile = classifier.classify(text)
        assert profile.evidence_source == profile.to_dict()["evidence_source"] == source
        assert classifier.legacy_build_profile(text) == [dict(profile.emotions), list(profile.emotional_keywords)]
        assert classifier.classify(text) is profile
    assert classifier.classify("I am happy").dominant_emotion == "joy"
    assert classifier.classify("I am not happy").dominant_emotion == "sadness"
    assert EmotionProfile.from_scores({"joy": 1}).evidence_source == "unavailable"
    assert classifier.classify_with_embedding("text", classifier.matrix[0]).evidence_source == "unavailable"


def test_default_factory_and_cli_select_telemetry(monkeypatch) -> None:
    assert inspect.signature(experiment.run_goemotions_experiment).parameters["system"].default == experiment.DEFAULT_SYSTEM
    system = experiment._system(
        experiment.DEFAULT_SYSTEM, GenerationConfig(model="fixture", provider="ollama"), ClassifierSettings()
    )
    assert isinstance(system, AffectTelemetrySystem)
    captured = {}
    monkeypatch.setattr(experiment, "run_goemotions_experiment", lambda **kwargs: captured.update(kwargs) or {})
    monkeypatch.setattr("sys.argv", ["goemotions", "--no-progress"])
    experiment.main()
    assert captured["system"] == "affect_telemetry_preserve_labels"


@pytest.mark.parametrize("corruption", ["hash", "baseline", "final"])
def test_same_response_comparison_rejects_mismatched_inputs(corruption) -> None:
    raw = '{"labels":["neutral"]}'
    metadata = {
        "raw_model_response": raw,
        "baseline_labels": ["neutral"],
        "final_scored_labels": ["neutral"],
        "label_policy_version": LABEL_POLICY_VERSION,
    }
    if corruption == "hash":
        metadata["raw_model_response_hash"] = "wrong"
    elif corruption == "baseline":
        metadata["baseline_labels"] = ["anger"]
    else:
        metadata["final_scored_labels"] = ["anger"]
    with pytest.raises(ValueError, match="GoEmotions"):
        experiment._classify_response(
            GoEmotionsExample(example_id="id", text="text", labels=frozenset()), list(LABELS), raw, metadata, "prompt"
        )


def test_parse_fallback_is_excluded_from_same_response_headline(monkeypatch) -> None:
    records = []
    for index, raw in enumerate(('{"labels":["neutral"]}', "not JSON")):
        monkeypatch.setattr(
            "llm.llm_client.LLMClient.chat", lambda self, prompt, output=raw, **kwargs: LLMResponse(text=output, raw={})
        )
        response, metadata = AffectTelemetrySystem(
            GenerationConfig(model="fixture", provider="ollama")
        ).predict("angry", list(LABELS))
        records.append(experiment._classify_response(
            GoEmotionsExample(example_id=str(index), text="angry", labels=frozenset({"neutral"})),
            list(LABELS), response, metadata, "prompt"
        ))
    metrics = experiment._metrics(records, list(LABELS), seed=13, bootstrap_samples=2)
    assert metrics["parse_failure_count"] == metrics["baseline_parse_failure_count"] == 1
    headline = metrics["headline_affect_comparison"]
    assert headline["sample_count"] == headline["excluded_count"] == 1
    assert headline["baseline"] == headline["final"]
    assert headline["paired_bootstrap_sample_f1"]["delta"] == 0
    assert metrics["paired_outcome_groups"]["affect_outcomes"]["unrecovered_parse_failure"] == 1


def test_old_record_without_policy_fields_remains_readable() -> None:
    record = {
        "id": "legacy", "gold_labels": ["anger"], "predicted_labels": ["anger", "neutral"],
        "baseline_labels": ["neutral"], "baseline_parse_error": None, "system_family": "bounded_affect_decision",
        "prima_decision": {"reason": "add_prima_lexical_emotion"}, "raw_response": '{"labels":["anger","neutral"]}',
    }
    metrics = experiment._metrics([record], list(LABELS), 13, 2)
    assert metrics["headline_affect_comparison"]["sample_count"] == 0
    assert record["prima_decision"]["reason"] == "add_prima_lexical_emotion"
    assert "label_policy_version" not in record


def test_default_experiment_persists_audit_and_resumes_without_inference(monkeypatch, tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "goemotions_val.json").write_text(json.dumps([{"id": "val", "text": "validation", "labels": [27]}]))
    dataset = data / "goemotions_test.json"
    dataset.write_text(json.dumps([
        {"id": "valid", "text": "I am angry.", "labels": [2]},
        {"id": "invalid", "text": "malformed", "labels": [27]},
    ]))
    calls = []

    def classify(self, prompt, **kwargs):
        calls.append(prompt)
        return LLMResponse(text='{"labels":["neutral"]}' if "I am angry." in prompt else "not JSON", raw={})

    monkeypatch.setattr("llm.llm_client.LLMClient.chat", classify)
    monkeypatch.setattr(experiment, "render_figures", lambda *args: None)
    options = {
        "dataset_path": dataset, "output_path": tmp_path / "out", "model": "fixture", "progress": False,
        "bootstrap_samples": 2, "llm_client": LLMClient(provider_name="ollama"),
    }
    result = experiment.run_goemotions_experiment(**options)
    assert result["system"] == "affect_telemetry_preserve_labels"
    assert result["samples"] == 2 and result["failed"] == 0
    assert result["metrics"]["parse_failure_count"] == 1
    assert result["metrics"]["headline_affect_comparison"]["sample_count"] == 1
    responses = json.loads(Path(result["artifacts"]["raw"]).read_text())["responses"]
    assert [record["id"] for record in responses] == ["valid", "invalid"]
    assert responses[0]["baseline_labels"] == responses[0]["predicted_labels"] == ["neutral"]
    assert responses[1]["raw_response"] == "not JSON" and responses[1]["parse_error"]
    assert responses[1]["parse_recovery"]["scope"] == "runtime_only_not_scored"
    for record in responses:
        assert record["response_hash"] == record["raw_model_response_hash"]
        assert record["label_policy_version"] == LABEL_POLICY_VERSION
    manifest = json.loads(Path(result["artifacts"]["manifest"]).read_text())
    assert manifest["benchmark_config"]["label_policy_version"] == LABEL_POLICY_VERSION
    assert manifest["benchmark_config"]["legacy_correction_enabled"] is False
    resumed = experiment.run_goemotions_experiment(**options, resume=True)
    assert len(calls) == 2
    assert json.loads(Path(resumed["artifacts"]["raw"]).read_text())["responses"] == responses
    assert resumed["metrics"]["headline_affect_comparison"] == result["metrics"]["headline_affect_comparison"]
