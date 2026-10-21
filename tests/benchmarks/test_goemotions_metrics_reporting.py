"""B2 statistic equivalence, probability eligibility, measurement and old/new reports."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from affect.emotion_prediction import EmotionPrediction
from affect.taxonomies.goemotions import LABELS
from benchmarks.campaign.config import ProviderConfig
from benchmarks.campaign.provider_session import SharedProviderSession
from benchmarks.common.artifacts import atomic_write_jsonl, read_jsonl
from benchmarks.common.reporting import classification_report, locomo_report
from benchmarks.common.timing import AttemptTimer
from benchmarks.goemotions.dataset import GoEmotionsExample
from benchmarks.goemotions.experiment import _classify_response, _metrics, run_goemotions_experiment
from benchmarks.goemotions.metrics import (
    correction_metrics,
    paired_bootstrap_sample_f1,
    performance_metrics,
    probability_eligibility,
)
from benchmarks.goemotions.systems import AffectTelemetrySystem, ClassifierSettings, EncoderSystem
from benchmarks.goemotions.training.calibration import write_calibration
from llm.generation_config import GenerationConfig, StructuredOutputMode
from llm.provider import ProviderError

FIXTURES = Path(__file__).parents[1] / "fixtures"
reference_spec = importlib.util.spec_from_file_location("bootstrap_reference", FIXTURES / "goemotions_review/bootstrap_reference.py")
reference = importlib.util.module_from_spec(reference_spec)
reference_spec.loader.exec_module(reference)


@pytest.mark.parametrize("gold,before,after", [
    ([], [], []), ([set()], [set()], [set()]),
    ([{"joy"}], [{"joy"}], [{"joy"}]), ([{"joy"}], [{"fear"}], [{"anger"}]),
    ([{"joy"}], [set()], [{"joy"}]),
    ([{"joy", "neutral"}, {"fear"}, set()], [{"joy"}, set(), set()], [{"neutral"}, {"fear"}, {"anger"}]),
])
@pytest.mark.parametrize("samples", [-1, 0, 1, 73])
def test_bootstrap_preserves_seeded_statistic_and_edge_cases(gold, before, after, samples):
    gold, before, after = ([frozenset(row) for row in values] for values in (gold, before, after))
    expected = reference.old_bootstrap(gold, before, after, samples=samples, seed=7)
    assert paired_bootstrap_sample_f1(gold, before, after, samples=samples, seed=7) == pytest.approx(expected, rel=0, abs=1e-14)
    swapped = paired_bootstrap_sample_f1(gold, after, before, samples=samples, seed=7)
    assert swapped["delta"] == pytest.approx(-expected["delta"], rel=0, abs=1e-14)
    assert swapped == pytest.approx(reference.old_bootstrap(gold, after, before, samples=samples, seed=7), rel=0, abs=1e-14)


def test_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        paired_bootstrap_sample_f1([frozenset()], [], [])


def calibrated_record():
    return {
        "id": "calibrated", "gold_labels": ["joy"], "predicted_labels": ["joy"], "parse_error": None,
        "probabilities_available": True,
        "prediction": {"probabilities": {label: 0.8 if label == "joy" else 0.1 for label in LABELS}, "model_id": "trained", "model_revision": "b" * 40},
        "probability_provenance": {"source": "trained_probability", "calibrated": True, "method": "scalar_temperature", "split": "dev", "dev_dataset_sha256": "d" * 64, "artifact_sha256": "a" * 64, "model_id": "trained", "model_revision": "b" * 40},
    }


@pytest.mark.parametrize("invalid", ["label_only", "raw_sigmoid", "partial", "nan", "range", "bool", "method", "artifact", "model", "revision", "dev", "parse"])
def test_ineligible_scores_never_become_confidence(invalid):
    row = calibrated_record()
    if invalid == "label_only":
        row["probabilities_available"] = False
    elif invalid == "raw_sigmoid":
        row["probability_provenance"]["calibrated"] = False
    elif invalid == "partial":
        row["prediction"]["probabilities"].pop("neutral")
    elif invalid in {"nan", "range", "bool"}:
        row["prediction"]["probabilities"]["joy"] = {"nan": float("nan"), "range": 1.01, "bool": True}[invalid]
    elif invalid == "parse":
        row["parse_error"] = "bad output"
    else:
        field = {"method": "method", "artifact": "artifact_sha256", "model": "model_id", "revision": "model_revision", "dev": "dev_dataset_sha256"}[invalid]
        row["probability_provenance"][field] = ""
    result = probability_eligibility(row, "d" * 64)
    assert not result["available"] and result["confidence"] is None
    metrics = _metrics([row], list(LABELS), 13, 2, expected_dev_hash="d" * 64)
    assert not metrics["probability_metrics"]["available"] and metrics["confidence"]["mean"] is None


def test_calibration_requires_every_selected_case_and_one_identity():
    row = calibrated_record()
    assert probability_eligibility(row, "d" * 64)["confidence"] == 0.8
    assert _metrics([row], list(LABELS), 13, 2, expected_dev_hash="d" * 64)["probability_metrics"]["available"]
    metrics = _metrics([row], list(LABELS), 13, 2, failures=[{"execution_failed": True}], expected_dev_hash="d" * 64)
    assert metrics["probability_metrics"]["selected_count"] == 2
    assert not metrics["probability_metrics"]["available"]
    mixed = copy.deepcopy(row)
    mixed["probability_provenance"]["artifact_sha256"] = "c" * 64
    assert not _metrics([row, mixed], list(LABELS), 13, 2, expected_dev_hash="d" * 64)["probability_metrics"]["available"]
    assert not _metrics([], list(LABELS), 13, 2)["probability_metrics"]["available"]


def test_partial_correction_harm_gain_and_added_label_precision():
    metrics = correction_metrics(
        [frozenset({"joy", "neutral"}), frozenset({"joy"})],
        [frozenset({"fear"}), frozenset({"joy"})],
        [frozenset({"joy"}), frozenset({"joy", "anger"})],
    )
    assert metrics["partial_f1_gains"] == metrics["partial_f1_harms"] == 1
    assert metrics["added_label_precision"] == 0.5
    assert metrics["improved_after_label_correction"] == 0
    assert metrics["regressed_after_label_correction"] == 1
    assert correction_metrics([], [], [])["added_label_precision"] is None


def test_cache_retries_tokens_and_provider_calls_are_measured():
    session = SharedProviderSession(ProviderConfig(kind="fake", model="fixture", revision="fixture-v1", context_window=4096), 1)
    generation = GenerationConfig(model="fixture", provider="fake", structured_output=StructuredOutputMode.JSON_SCHEMA, retries=1)
    delegate = session.provider.provider

    class Flaky:
        capabilities = delegate.capabilities
        name = "fake"
        failed = False

        def validate(self, request):
            delegate.validate(request)

        def send(self, request):
            if not self.failed:
                self.failed = True
                raise ProviderError("timeout")
            return delegate.send(request)

    session.provider.provider = Flaky()
    system = AffectTelemetrySystem(generation).use_client(session.client)
    records = []
    for index in range(2):
        raw, metadata = system.predict("same text", list(LABELS))
        assert metadata["prediction"]["confidence"] is None
        record = _classify_response(GoEmotionsExample("same text", frozenset({"joy"}), str(index)), list(LABELS), raw, metadata, "prompt", 1000)
        records.append(record)
    assert records[0]["classification_provider"]["attempts"] == 2
    assert records[0]["classification_provider"]["retries"] == 1
    assert records[1]["classification_provider"]["attempts"] == 0
    assert records[1]["classification_provider"]["cache_hits"] == 1
    measured = performance_metrics(records, {"complete": True, "active_wall_seconds": 1})
    assert measured["classification_provider_calls"] == 2
    assert measured["successful_uncached_requests"] == 1 and measured["response_cache_hits"] == 1
    assert measured["wall_throughput_examples_per_second"] == 2
    assert measured["examples_per_summed_latency_second"] == 1
    assert measured["actual_uncached_tokens"] is None  # Failed requests do not reveal billed tokens.
    assert measured["logical_tokens"] == 2 * session.telemetry()["total_tokens"]
    assert classification_report(records)["runtime_reflection"]["executed"] == 0


def test_monotonic_active_attempts_exclude_calendar_downtime_and_open_tail(monkeypatch, tmp_path):
    clock = {"now": 0}
    monkeypatch.setattr("benchmarks.common.timing.time.perf_counter", lambda: clock["now"])
    path = tmp_path / "timings.jsonl"
    first = AttemptTimer(path)
    clock["now"] = 3
    first.record("inference", finished=True)
    rows = read_jsonl(path)
    for row in rows:
        row["observed_at"] = "2010-01-01T00:00:00+00:00"
    atomic_write_jsonl(path, rows)
    clock["now"] = 100
    second = AttemptTimer(path)
    assert not second.summary()["complete"]
    assert performance_metrics([], second.summary())["wall_throughput_examples_per_second"] is None
    clock["now"] = 105
    second.record("inference", finished=True)
    summary = second.summary()
    assert summary["active_wall_seconds"] == 8 and summary["attempt_count"] == 2
    assert summary["calendar_seconds"] > 10 * 365 * 86400
    assert sum(summary["phase_seconds"].values()) == 8


def test_failures_are_disjoint_and_old_confidence_is_unavailable():
    records = [
        {"gold_labels": ["joy"], "predicted_labels": ["joy"], "parse_error": None},
        {"gold_labels": ["joy"], "predicted_labels": ["neutral"], "parse_error": None, "prediction": {"confidence": 1}},
        {"gold_labels": ["joy"], "predicted_labels": [], "parse_error": "malformed"},
    ]
    metrics = _metrics(records, list(LABELS), 13, 2, failures=[{"execution_failed": True}])
    assert metrics["primary_outcome_counts"] == {"execution_failure": 1, "parse_failure": 1, "label_error": 1, "exact_match": 1}
    assert sum(metrics["primary_outcome_counts"].values()) == metrics["populations"]["selected"] == 4
    old = classification_report(records)
    assert old["confidence"]["mean"] is None
    assert old["classification_provider_calls"] is None
    assert performance_metrics(records)["wall_throughput_examples_per_second"] is None


def test_committed_person1_old_and_new_contracts_keep_stage_and_scope_distinctions():
    fixture = json.loads((FIXTURES / "campaign/person1_report_contract.json").read_text())
    old, new = (entry["record"] for entry in fixture)
    assert locomo_report([old])["evidence"]["candidate"]["mean_recall"] is None
    summary = locomo_report([old, new])
    assert summary["evidence"]["candidate"]["mean_recall"] == 1
    assert summary["evidence"]["accumulated"]["mean_recall"] == 1
    assert summary["evidence"]["supplied"]["mean_recall"] == 0
    assert not summary["diagnostic_flags_additive"]
    assert summary["primary_failure_counts"] == {"correct": 1}
    assert summary["qa_count"] == 2 and summary["conversation_replay"] is None
    assert summary["answer_scores"][0] is None
    producer = json.loads((FIXTURES / "campaign/person1_goal4_producer_contract.json").read_text())
    qa = {"schema_version": "2.0", "reflection_audit": producer["qa"]["reflection_audit"]}
    assert locomo_report([qa])["reflection_audits"] == [producer["qa"]["reflection_audit"]]
    replay = [{"conversation_id": "conv", "historical_turn_count": 2, "admitted_turn_count": 1, "maintenance_complete": True}]
    summary = locomo_report([new, new], replay)
    assert summary["qa_count"] == 2
    assert summary["conversation_replay_coverage"]["historical_turns"] == 2
    assert summary["conversation_replay_coverage"]["admitted_turns"] == 1
    assert summary["qa_component_coverage"]["executed_counts"] is None


def test_native_campaign_reports_cache_confidence_failures_and_active_resume(monkeypatch, tmp_path):
    dataset = tmp_path / "goemotions_test.json"
    dataset.write_text(json.dumps([
        {"id": "one", "text": "same", "labels": [17]},
        {"id": "two", "text": "same", "labels": [27]},
    ]))
    dataset.with_name("goemotions_val.json").write_text(json.dumps([{"id": "dev", "text": "dev", "labels": [27]}]))
    monkeypatch.setattr("benchmarks.goemotions.experiment.render_figures", lambda *args: None)
    session = SharedProviderSession(ProviderConfig(kind="fake", model="fixture", revision="v1", context_window=4096), 1)
    options = {
        "dataset_path": dataset, "output_path": tmp_path / "out", "provider": "fake", "model": "fixture",
        "bootstrap_samples": 2, "llm_client": session.client, "progress": False,
    }
    first = run_goemotions_experiment(**options)
    metrics = first["metrics"]
    assert metrics["primary_outcome_counts"] == {"execution_failure": 0, "parse_failure": 0, "label_error": 1, "exact_match": 1}
    assert metrics["confidence"]["mean"] is None
    measured = metrics["performance"]
    assert measured["classification_provider_calls"] == measured["successful_uncached_requests"] == measured["response_cache_hits"] == 1
    assert measured["logical_classification_provider_calls"] == 2
    assert measured["logical_tokens"] == 2 * measured["actual_uncached_tokens"]
    assert measured["wall_throughput_examples_per_second"] == 2 / metrics["timing"]["active_wall_seconds"]
    report = Path(first["artifacts"]["report"]).read_text(encoding="utf-8")
    assert "Gold support" in report and "| remorse |" in report
    assert '"label_error": 1' in report and '"mean": null' in report
    second = run_goemotions_experiment(**options, resume=True)
    assert second["metrics"]["timing"]["attempt_count"] == 2
    assert second["metrics"]["timing"]["active_wall_seconds"] >= metrics["timing"]["active_wall_seconds"]
    assert second["metrics"]["performance"]["classification_provider_calls"] == 1
    assert session.telemetry()["request_attempts"] == 1


def test_calibrated_serialization_protects_case_identity_and_exposes_verified_confidence():
    metadata = calibrated_record()
    metadata.update(id="wrong", gold_labels=["fear"])
    record = _classify_response(
        GoEmotionsExample("comment", frozenset({"joy"}), "actual"), list(LABELS), '{"labels":["joy"]}', metadata,
        "prompt", expected_dev_hash="d" * 64,
    )
    assert record["id"] == "actual" and record["gold_labels"] == ["joy"]
    assert record["confidence_available"] and record["confidence"] == record["prediction"]["confidence"] == 0.8


def test_encoder_calibration_provenance_uses_actual_artifact_and_model_identity(tmp_path):
    path = tmp_path / "calibration.json"
    revision = "b" * 40
    write_calibration(path, {"method": "scalar_temperature", "temperature": 1.2}, model_id="trained", dev_hash="d" * 64, model_revision=revision)
    system = EncoderSystem(
        GenerationConfig(model="trained", provider="ollama", revision=revision), "trained_encoder", "trained_encoder",
        ClassifierSettings(calibration_path=path),
    )
    assert system._encoder.revision == revision
    prediction = EmotionPrediction.from_probabilities(
        calibrated_record()["prediction"]["probabilities"], {}, "trained", revision,
        metadata={"calibration_temperature": 1.2},
    )
    provenance = system._calibration_provenance(prediction)
    assert provenance["calibrated"] and provenance["model_revision"] == revision
    assert len(provenance["artifact_sha256"]) == 64
    write_calibration(path, {"method": "scalar_temperature", "temperature": 1.2}, model_id="trained", dev_hash="d" * 64)
    assert not system._calibration_provenance(prediction)["calibrated"]
    with pytest.raises(ValueError, match="immutable"):
        write_calibration(path, {}, model_id="trained", dev_hash="d" * 64, model_revision="main")


def test_legacy_parse_recovery_stays_a_parse_failure_without_rewriting_prediction():
    record = {
        "gold_labels": ["neutral"], "predicted_labels": ["neutral"], "baseline_labels": [],
        "baseline_parse_error": "malformed original", "parse_error": None,
        "raw_model_response": "not JSON", "raw_response": '{"labels":["neutral"]}',
        "system_family": "bounded_affect_decision",
    }
    metrics = _metrics([record], list(LABELS), 13, 2)
    assert metrics["primary_outcome_counts"]["parse_failure"] == 1
    assert metrics["primary_outcome_counts"]["exact_match"] == 0
    assert metrics["headline_affect_comparison"]["sample_count"] == 0
    assert classification_report([record])["primary_outcome_counts"]["parse_failure"] == 1
    assert record["predicted_labels"] == ["neutral"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, "1"])
def test_timing_rejects_invalid_saved_observations(tmp_path, bad):
    timer = AttemptTimer(tmp_path / "timing.jsonl")
    rows = read_jsonl(timer.path)
    rows[0]["elapsed_seconds"] = bad
    atomic_write_jsonl(timer.path, rows)
    with pytest.raises(ValueError, match="duration"):
        timer.summary()


def test_missing_old_labels_are_unscored_not_exact_success():
    report = classification_report([{"prediction": {"confidence": 1.0}}])
    assert report["primary_outcome_counts"]["unscored"] == 1
    assert report["primary_outcome_counts"]["exact_match"] == 0
    assert report["confidence"]["mean"] is None
    assert report["label_correction"]["mean_sample_f1_delta"] is None


def test_serializer_respects_explicit_unpaired_exclusion():
    record = _classify_response(
        GoEmotionsExample("comment", frozenset({"joy"}), "id"), list(LABELS), '{"labels":["joy"]}',
        {"baseline_labels": ["joy"], "same_response_comparison_eligible": False}, "prompt",
    )
    assert record["same_response_comparison_eligible"] is False
    assert not classification_report([record])["label_correction"]["available"]
