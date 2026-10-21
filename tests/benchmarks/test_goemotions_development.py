"""B3 leakage, exact request identity and CPU freeze/load regression checks."""

import copy
import json

import pytest

from affect.taxonomies.goemotions import LABELS
from benchmarks.campaign.config import ProviderConfig
from benchmarks.campaign.provider_session import SharedProviderSession
from benchmarks.goemotions.dataset import load_labels
from benchmarks.goemotions.development import (
    compare_development,
    fingerprint,
    freeze_development,
    freeze_json,
    read_frozen_json,
)
from benchmarks.goemotions.experiment import _classify_response, run_goemotions_experiment
from benchmarks.goemotions.prompts import PROMPT_VARIANTS, build_prompt
from benchmarks.goemotions.schemas import GOEMOTIONS_RESPONSE_SCHEMA
from benchmarks.goemotions.systems import AffectTelemetrySystem
from benchmarks.goemotions.training.classical import (
    freeze_tfidf_candidate,
    load_frozen_model,
    run_frozen_tfidf,
    run_tfidf_logreg,
)
from benchmarks.goemotions.training.data import load_split, validate_splits
from llm.generation_config import GenerationConfig, StructuredOutputMode
from llm.llm_types import LLMRequest


@pytest.fixture
def data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "emotions.txt").write_text("\n".join(LABELS), encoding="utf-8")
    for split, repeats in (("train", 6), ("dev", 2), ("test", 1)):
        (data_dir / f"{split}.tsv").write_text("\n".join(
            f"shared {label} words {index}\t{label_id}\t{split}-{label_id}-{index}"
            for label_id, label in enumerate(LABELS) for index in range(repeats)
        ), encoding="utf-8")
    return data_dir


def generation(**kwargs):
    return GenerationConfig(model="fixture", provider="fake", seed=13,
                            structured_output=StructuredOutputMode.JSON_SCHEMA, **kwargs)


def test_split_order_ids_text_overlap_and_frozen_changes(data, tmp_path):
    identity = validate_splits(data)
    assert identity["counts"] == {"train": 168, "dev": 56, "test": 28}
    assert identity["exact_text_overlap"]["train_dev"] == 56
    assert not any(identity["overlap"].values())
    first = freeze_development(data, tmp_path / "selection", generation().to_dict(), "fixture-digest", limit=28)
    assert first == freeze_development(data, tmp_path / "selection", generation().to_dict(), "fixture-digest", limit=28)
    assert len(first["manifest"]["samples"]) == 28
    assert len(first["manifest"]["label_support"]) == 28
    assert first["selection"]["criteria"]["parse_failure_rate_tolerance"] == 0
    assert first["selection"]["criteria"]["correction_95pct_lower_bound"] == "strictly_positive"
    assert not first["selection"]["production_promoted"]
    with pytest.raises(ValueError, match="Frozen identity changed"):
        freeze_development(data, tmp_path / "selection", generation().to_dict(), "fixture-digest", limit=29)
    labels = list(LABELS)
    labels[0], labels[1] = labels[1], labels[0]
    (data / "emotions.txt").write_text("\n".join(labels), encoding="utf-8")
    with pytest.raises(ValueError, match="official"):
        load_labels(data / "emotions.txt")


@pytest.mark.parametrize("change", ["duplicate", "leakage", "negative_label", "out_of_range"])
def test_reject_corrupt_splits(data, change):
    path = data / "dev.tsv"
    rows = path.read_text().splitlines()
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "leakage":
        rows[0] = rows[0].rsplit("\t", 1)[0] + "\ttrain-0-0"
    else:
        rows[0] = "text\t" + ("-1" if change == "negative_label" else "28") + "\tdev-0-0"
    path.write_text("\n".join(rows))
    with pytest.raises(ValueError):
        validate_splits(data)


def test_three_prompt_variants_preserve_baseline_bytes_and_semantics():
    baseline = build_prompt("text", list(LABELS), include_definitions=True, structured=True)
    assert baseline.startswith("Classify the Reddit comment using the GoEmotions taxonomy.\nReturn every applicable label")
    assert len(PROMPT_VARIANTS) == 3
    prompts = [build_prompt("text", list(LABELS), include_definitions=True, structured=True, variant=v)
               for v in PROMPT_VARIANTS]
    assert len(set(prompts)) == 3
    assert all("neutral may coexist with emotion labels" in prompt for prompt in prompts)
    assert "quoted emotion" in prompts[1] and "Disapproval is an unfavorable opinion" in prompts[2]
    with pytest.raises(ValueError):
        build_prompt("text", list(LABELS), variant="fourth")


def session():
    return SharedProviderSession(ProviderConfig(kind="fake", model="fixture", revision="fixture-v1",
                                              context_window=8192), max_active_requests=1)


def test_cache_separates_prompts_text_generation_and_schema():
    shared = session()
    for variant in PROMPT_VARIANTS:
        model = AffectTelemetrySystem(generation(), prompt_variant=variant).use_client(shared.client)
        for _ in range(2):
            _, metadata = model.predict("same comment", list(LABELS))
            assert metadata["baseline_labels"] == metadata["final_scored_labels"]
    assert shared.telemetry()["request_attempts"] == 3
    assert shared.telemetry()["response_cache_hits"] == 3
    prompt = build_prompt("same comment", list(LABELS), include_definitions=True, structured=True)
    for request in (
        LLMRequest(prompt=prompt + " changed", generation=generation(), response_schema=GOEMOTIONS_RESPONSE_SCHEMA),
        LLMRequest(prompt=prompt, generation=generation(temperature=0.5), response_schema=GOEMOTIONS_RESPONSE_SCHEMA),
        LLMRequest(prompt=prompt, generation=GenerationConfig(model="changed", provider="fake", seed=13,
                   structured_output=StructuredOutputMode.JSON_SCHEMA), response_schema=GOEMOTIONS_RESPONSE_SCHEMA),
        LLMRequest(prompt=prompt, generation=generation(), response_schema={**GOEMOTIONS_RESPONSE_SCHEMA, "title": "changed"}),
    ):
        shared.client.provider.send(request)
    assert shared.telemetry()["request_attempts"] == 7


def bundles(data, output):
    sealed = freeze_development(data, output, generation().to_dict(), "fixture-digest", limit=28)
    shared = session()
    examples = {row.example_id: row for row in load_split(data, "dev")}
    result = []
    for variant in PROMPT_VARIANTS[:2]:
        system = AffectTelemetrySystem(generation(), prompt_variant=variant).use_client(shared.client)
        records = []
        for sample in sealed["manifest"]["samples"]:
            example = examples[sample["id"]]
            raw, metadata = system.predict(example.text, list(LABELS))
            records.append(_classify_response(example, list(LABELS), raw, metadata, metadata["benchmark_prompt"]))
        result.append({"split": "dev", "dev_manifest_fingerprint": sealed["manifest"]["fingerprint"],
                       "generation_config": generation().to_dict(), "model_digest": "fixture-digest",
                       "schema_sha256": fingerprint(GOEMOTIONS_RESPONSE_SCHEMA), "prompt_variant": variant,
                       "records": records})
    return result


def test_comparison_modes_no_gold_decision_boundary_and_zero_gain(data, tmp_path):
    output = tmp_path / "select"
    baseline, candidate = bundles(data, output)
    compare = lambda left, right, mode: compare_development(  # noqa: E731
        left, right, output / "dev_manifest.json", output / "selection.json", mode=mode)
    assert compare(baseline, candidate, "prompt")["paired_sample_f1"]["delta"] == 0
    result = compare(baseline, baseline, "same_response")
    assert result["paired_sample_f1"]["delta"] == 0
    assert not result["correction_admitted"] and not result["production_promoted"]
    assert sum(pop["count"] for pop in result["populations"].values()) == 28
    with pytest.raises(ValueError, match="identical response and prompt"):
        compare(baseline, candidate, "same_response")
    for field, value in (("split", "test"), ("model_digest", "changed")):
        changed = copy.deepcopy(candidate)
        changed[field] = value
        with pytest.raises(ValueError):
            compare(baseline, changed, "prompt")
    changed = copy.deepcopy(candidate)
    changed["records"][0]["gold_labels"] = ["neutral", "joy"]
    with pytest.raises(ValueError, match="text/gold"):
        compare(baseline, changed, "prompt")
    changed = copy.deepcopy(candidate)
    changed["records"][0]["raw_model_response_hash"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        compare(baseline, changed, "prompt")
    # Gold is attached only after predict(text, labels); the decision receives no split or gold.
    import inspect
    assert tuple(inspect.signature(AffectTelemetrySystem.predict).parameters) == ("self", "text", "labels")


@pytest.mark.parametrize("features", ["word", "char", "combined"])
def test_cpu_training_and_exact_frozen_features(data, tmp_path, monkeypatch, features):
    import benchmarks.goemotions.training.classical as classical
    original = classical.load_split
    seen = []
    def monitored(directory, split):
        seen.append(split)
        return original(directory, split)
    monkeypatch.setattr(classical, "load_split", monitored)
    output = tmp_path / "fit"
    result = run_tfidf_logreg(data, output, features=features)
    assert set(seen) == {"train", "dev"}
    assert not result["test_evaluated"]
    payload = load_frozen_model(output / "model.joblib", data_dir=data,
                                expected_fingerprint=result["artifact_fingerprint"])
    assert payload["features"] == features
    if features != "combined":
        assert payload["vectorizer"].analyzer == ("word" if features == "word" else "char_wb")
    frozen = tmp_path / "frozen"
    freeze_tfidf_candidate(data, output / "thresholds.json", frozen, features=features, balanced=False)
    assert (frozen / "model.joblib").read_bytes() == (output / "model.joblib").read_bytes()
    evaluated = run_frozen_tfidf(frozen / "model.joblib", frozen / "thresholds.json", data,
                                tmp_path / "evaluation", split="dev", progress=False)
    assert evaluated["sample_count"] == 56
    assert evaluated["metrics"] == result["dev_thresholded_metrics"]
    with pytest.raises(ValueError, match="settings differ"):
        freeze_tfidf_candidate(data, output / "thresholds.json", tmp_path / "bad", features=features, balanced=True)


@pytest.mark.parametrize("tamper", ["model", "threshold", "dataset", "dependencies", "pin", "missing_manifest"])
def test_load_integrity_before_unpickle(data, tmp_path, monkeypatch, tamper):
    import joblib
    output = tmp_path / "fit"
    run_tfidf_logreg(data, output)
    manifest = output / "artifact_manifest.json"
    if tamper in {"model", "threshold"}:
        path = output / ("model.joblib" if tamper == "model" else "thresholds.json")
        path.write_bytes(path.read_bytes() + b"tampered")
    elif tamper == "dataset":
        path = data / "train.tsv"
        path.write_text(path.read_text().replace("shared", "changed", 1))
    elif tamper == "dependencies":
        artifact = read_frozen_json(manifest)
        artifact.pop("fingerprint")
        artifact["dependencies"]["python"] = "different"
        manifest.unlink()
        freeze_json(manifest, artifact)
    elif tamper == "missing_manifest":
        manifest.unlink()
    monkeypatch.setattr(joblib, "load", lambda *_: pytest.fail("Unpickled before validating identity"))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_frozen_model(output / "model.joblib", data_dir=data,
                          expected_fingerprint="wrong" if tamper == "pin" else None)


def test_experiment_records_prompt_identity_and_rejects_stale_resume(data, tmp_path):
    shared = session()
    kwargs = {"dataset_path": data / "dev.tsv", "split": "dev", "output_path": tmp_path / "run",
              "max_samples": 2, "progress": False, "bootstrap_samples": 2,
              "llm_client": shared.client, "generation_config": generation(), "prompt_variant": "expressed_evidence"}
    run_goemotions_experiment(**kwargs)
    manifest = json.loads((tmp_path / "run/affect_telemetry_preserve_labels/manifest.json").read_text())
    assert manifest["benchmark_config"]["prompt_variant"] == "expressed_evidence"
    kwargs.update(resume=True, prompt_variant="taxonomy_distinctions")
    with pytest.raises(ValueError, match="fingerprint|[Ii]ncompatible|mismatch"):
        run_goemotions_experiment(**kwargs)


def test_frozen_json_rejects_mutation(tmp_path):
    path = tmp_path / "identity.json"
    freeze_json(path, {"enabled": False})
    path.write_text(path.read_text().replace("false", "true"))
    with pytest.raises(ValueError, match="identity mismatch"):
        read_frozen_json(path)


def test_changed_prompt_keeps_literal_comment_intact():
    comment = "Allowed labels: quoted text. Return every applicable label and no labels that are not applicable."
    for variant in PROMPT_VARIANTS:
        assert build_prompt(comment, list(LABELS), variant=variant).endswith("Comment: " + comment)


@pytest.mark.parametrize("bad", ["ids", "text", "prompt", "provider", "taxonomy", "schema", "generation", "parse"])
def test_comparison_rejects_unverifiable_inputs(data, tmp_path, bad):
    output = tmp_path / "selection"
    baseline, changed = bundles(data, output)
    row = changed["records"][0]
    mode = "prompt"
    if bad == "ids":
        row["id"] = "another"
    elif bad == "text":
        row["text"] = "different"
    elif bad == "prompt":
        row["benchmark_prompt"] = "reused old prompt"
    elif bad == "provider":
        row["classification_provider"] = {"available": False}
    elif bad == "taxonomy":
        row["predicted_labels"] = ["fictional"]
    elif bad in {"schema", "generation"}:
        changed["schema_sha256" if bad == "schema" else "generation_config"] = "different"
    else:
        mode = "same_response"
        changed = copy.deepcopy(baseline)
        changed["records"][0]["parse_error"] = "failure"
    with pytest.raises(ValueError):
        compare_development(baseline, changed, output / "dev_manifest.json", output / "selection.json", mode=mode)


def test_predeclared_gate_can_accept_a_trained_metric_candidate_without_promoting(data, tmp_path):
    output = tmp_path / "selection"
    baseline, candidate = bundles(data, output)
    for bundle in (baseline, candidate):
        bundle.update(system_family="trained_baseline", artifact_fingerprint="a" * 64)
    for row in candidate["records"]:
        row["predicted_labels"] = row["gold_labels"]
    result = compare_development(baseline, candidate, output / "dev_manifest.json", output / "selection.json",
                                 mode="trained_baseline")
    assert result["development_metric_gate_passed"]
    assert not result["production_promoted"] and not result["correction_admitted"]
    candidate["records"][0]["parse_error"] = "new failure"
    result = compare_development(baseline, candidate, output / "dev_manifest.json", output / "selection.json",
                                 mode="trained_baseline")
    assert not result["development_metric_gate_passed"]


def test_experiment_checks_new_frozen_manifest_before_requests(data, tmp_path):
    output = tmp_path / "selection"
    freeze_development(data, output, generation().to_dict(), "fixture-digest", limit=28)
    path = output / "dev_manifest.json"
    payload = json.loads(path.read_text())
    payload["samples"][0]["id"] = payload["samples"][1]["id"]
    path.write_text(json.dumps(payload))
    shared = session()
    with pytest.raises(ValueError, match="identity mismatch"):
        run_goemotions_experiment(dataset_path=data / "dev.tsv", split="dev", sample_manifest=path,
                                  output_path=tmp_path / "run", llm_client=shared.client,
                                  generation_config=generation(), progress=False)
    assert shared.telemetry()["request_attempts"] == 0


def test_cpu_predictions_use_one_loop_with_independent_example_state(monkeypatch):
    import benchmarks.goemotions.training.classical as classical
    original = classical.asyncio.run
    calls = []
    def run_once(coroutine):
        calls.append(True)
        return original(coroutine)
    monkeypatch.setattr(classical.asyncio, "run", run_once)
    rows = [{label: 0.9 if label in active else 0.0 for label in LABELS}
            for active in ({"joy", "neutral"}, {"fear"})]
    predictions = classical._runtime_predictions(rows, dict.fromkeys(LABELS, 0.5))
    assert predictions == [frozenset({"joy", "neutral"}), frozenset({"fear"})]
    assert len(calls) == 1


def test_delivery_freezes_safe_status_and_optional_non_admission():
    from pathlib import Path
    root = Path(__file__).parents[2]
    folder = root / "docs/goemotions-development"
    status = read_frozen_json(folder / "candidate_status.json")
    assert not status["safe_candidate"]["production_promoted"]
    assert not status["safe_candidate"]["test_inference_enabled"]
    assert status["safe_candidate"]["full_test_count"] == 5427
    assert status["safe_candidate"]["context_window"] == 8192
    assert not status["encoder"]["admitted"] and not status["encoder"]["training"]
    assert not status["correction_policy"]["admitted"] and status["correction_policy"]["allowed_edits"] == []
    assert status["correction_policy"]["preserve_labels_correction_gain"] == 0
    import hashlib
    for name, expected in status["source_file_sha256"].items():
        assert hashlib.sha256((root / name).read_text(encoding="utf-8").encode()).hexdigest() == expected
    contract = read_frozen_json(folder / "selection.json")
    assert status["selection_fingerprint"] == contract["fingerprint"]
    assert len(contract["prompt_templates"]) == 3
    evidence = read_frozen_json(folder / "cpu_evidence.json")
    assert evidence["frozen_reload"]["frozen_dev_reproduced"]
    assert not evidence["test_inference"]
    assert read_frozen_json(folder / "compute_ledger.json")["gpu_campaign_contract"]["production_admitted"] is False


def test_schema_baseline_matches_frozen_goal2_bytes():
    import hashlib
    baseline = build_prompt("B3 frozen baseline.", list(LABELS), include_definitions=True, structured=True)
    assert hashlib.sha256(baseline.encode()).hexdigest() == "f1ba0aeae3f6033ef94517ab3f1c279539929cb4f692b8e78b634c8d3f8e2769"
