"""Actual pinned Person 1 producers exercised through Person 2 campaign consumers."""

import json
from pathlib import Path

import pytest

from benchmarks.campaign.config import CampaignConfig
from benchmarks.campaign.orchestrator import run_campaign
from benchmarks.campaign.provider_session import FakeProvider
from benchmarks.common.artifacts import read_json, read_jsonl
from benchmarks.common.bundle import create_bundle, materialize_bundle
from benchmarks.common.reporting import locomo_report
from config.runtime_config import RuntimeConfig
from llm.llm_types import LLMResponse

ROOT = Path(__file__).parents[2]
PRODUCER = ROOT / "tests/fixtures/locomo_review/goal4_final_candidates.json"
pytestmark = pytest.mark.skipif(not PRODUCER.is_file(), reason="Requires pinned Person 1 final producer in the combined checkout")


@pytest.mark.parametrize("variant", ["controlled_document_ingestion", "normal_prima_admission"])
def test_actual_replay_qa_contracts_reports_and_export(monkeypatch, tmp_path, variant):
    settings = json.loads(PRODUCER.read_text(encoding="utf-8"))["safe_baseline"]["runtime_config"]
    typed = RuntimeConfig.model_validate(settings)
    assert typed.reflection_enabled is False and typed.embedding_dimensions == 64
    calls = []
    def abstain(self, request):
        calls.append(request)
        return LLMResponse(text='{"answer":"","evidence":[],"insufficient_information":true}',
                           raw={"fake": True}, usage={"prompt_tokens": 2, "completion_tokens": 3}, provider="fake")
    monkeypatch.setattr(FakeProvider, "send", abstain)
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps([{"sample_id": "source-conversation", "conversation": {
        "speaker_a": "Ava", "speaker_b": "Maya", "session_1_date_time": "2023-08-20",
        "session_1": [{"speaker": "Maya", "dia_id": "D1:1", "text": "I live in Paris with my aunt."},
                      {"speaker": "Ava", "dia_id": "D1:2", "text": "Maya lives in Paris and visits her aunt."}]},
        "qa": [{"question": "Where does Maya live?", "answer": "Paris", "category": 1, "evidence": ["D1:1"]},
               {"question": "Who took Maya to Mars?", "answer": "No information available", "category": 5, "evidence": []}]}]))
    root = tmp_path / "campaign"
    from benchmarks.campaign.kaggle import prepare_run
    prepare_run(root, {"scope": "synthetic_actual_producer_contract"}, resume=False, fresh=True)
    payload = {"output_root": str(root), "provider": {"kind": "fake", "model": "fixture", "revision": "fake-v1", "context_window": 8192},
               "runtime": settings, "scheduler": {"cpu_workers": 1, "max_gpu_requests": 1},
               "benchmarks": [{"id": "locomo", "benchmark": "locomo", "dataset_path": str(data), "profile": "prima_full",
                   "variant": variant, "context_budget": 6144, "max_items": 2,
                   "options": {"max_conversations": 1, "parallel_workers": 1, "maintenance_mode": "disabled"}}]}
    result = run_campaign(CampaignConfig.model_validate(payload))
    assert result["status"] == "complete"
    if variant == "controlled_document_ingestion":
        assert calls, "The actual answer path must reach the injected offline provider"
    manifest = read_json(root / "manifest.json")
    child_path = root / manifest["modes"]["locomo"]["child_manifest"]
    child = read_json(child_path)
    assert child["benchmark_config"]["runtime_config"] == typed.model_dump(mode="json")
    assert child["benchmark_config"]["runtime_config_fingerprint"] == typed.fingerprint
    assert child["benchmark_config"]["embedding_identity"]["embedding_dimensions"] == 64
    rows = read_jsonl(child_path.parent / "checkpoints/records.jsonl")
    records = [row["prediction"]["diagnostics"]["locomo_record"] for row in rows]
    report = read_json(root / "aggregate.json")["report_semantics"]["locomo"]
    assert report["qa_count"] == 2 and report["new_schema_qa_count"] == 2
    assert report["conversation_replay_coverage"]["observed_conversations"] == 1
    assert report["conversation_replay_coverage"]["historical_turns"] == 2
    assert report["reflection"]["engine_executed"] == 0
    assert report["reflection"]["corrections"]["accepted"] == 0
    assert report["qa_memory_writes"] == 0
    assert report["answer_attempts"] == [row["answer_attempt"] for row in records]
    assert report["request_budgets"] == [row["request_budget"] for row in records]
    for stage in ("candidate", "accumulated", "supplied", "cited"):
        values = [row[f"{stage}_evidence_recall"] for row in records if row[f"{stage}_evidence_recall"] is not None]
        assert report["evidence"][stage]["mean_recall"] == pytest.approx(sum(values) / len(values))
    if variant == "controlled_document_ingestion":
        assert report["evidence"]["candidate"]["mean_recall"] > report["evidence"]["cited"]["mean_recall"]
    assert all(row["answer_scores"]["official_locomo_f1"] is not None for row in records)
    replay = read_json(child_path.parent / "raw/conversation_diagnostics.json")
    assert len(replay) == 1
    audits = [read_json(path) for path in (child_path.parent / "admission").glob("*.json")]
    assert len(audits) == 1 and audits[0]["turn_count"] == len(audits[0]["turns"]) == 2
    assert report["admission_audit"]["observed_conversations"] == 1
    assert report["admission_audit"]["turn_count"] == 2 and report["admission_audit"]["complete"]
    admission = report["admission_audit"]
    assert admission["accepted"] + admission["rejected"] + admission["unavailable_decisions"] == 2
    if variant == "normal_prima_admission":
        assert admission["unavailable_decisions"] == 0 and admission["reasons"]
    # Run the actual notebook consumers on these producer artifacts.
    namespace = {"json": json, "Path": Path, "STATE": {"run_root": str(root), "mode_id": "locomo"},
        "SELECTED_BENCHMARK": "locomo", "read_jsonl": read_jsonl}
    notebook = json.loads((ROOT / "PRIMA_Kaggle_Benchmark.ipynb").read_text(encoding="utf-8"))
    for index in (22, 24):
        exec("".join(notebook["cells"][index]["source"]), namespace)  # noqa: S102 - checked-in consumers on synthetic runtime output
    assert namespace["architecture_diagnostics"]["admission_audit"] == admission
    assert namespace["benchmark_evaluation"]["official_locomo_f1"] is not None
    old = locomo_report([{"prediction": "legacy", "failure_category": None}])
    assert old["evidence"]["candidate"]["mean_recall"] is None
    assert old["reflection"]["engine_executed"] is None
    archive = tmp_path / "producer-consumer.zip"
    assert create_bundle(root, archive)["valid"]
    recovered = tmp_path / "recovered"
    materialize_bundle(archive, recovered)
    assert read_json(recovered / "aggregate.json")["report_semantics"] == read_json(root / "aggregate.json")["report_semantics"]



def test_auxiliary_staging_is_pinned_local_cpu_and_does_not_claim_execution(tmp_path):
    import hashlib

    from benchmarks.campaign.preflight import verify_auxiliary_models
    from tests.benchmarks.test_campaign import _payload
    payload = _payload(tmp_path / "unused")
    payload["runtime"] = {"embedding_backend": "sentence_transformer", "embedding_dimensions": 384,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_revision": "a" * 40,
        "embedding_local_snapshot": str(tmp_path / "snapshot"), "embedding_device": "cpu"}
    config = CampaignConfig.model_validate(payload)
    with pytest.raises(FileNotFoundError):
        verify_auxiliary_models(config)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    content = b"{}"
    (snapshot / "config.json").write_bytes(content)
    (snapshot / "prima_model_manifest.json").write_text(json.dumps({
        "model_identifier": payload["runtime"]["embedding_model"], "revision": "a" * 40,
        "files": {"config.json": hashlib.sha256(content).hexdigest()}}))
    identity = verify_auxiliary_models(config)["embedding"]
    assert identity["device"] == "cpu" and identity["local_files_only"]
    assert not identity["execution_validated"]
    (snapshot / "config.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="hashes"):
        verify_auxiliary_models(config)
    payload["runtime"]["embedding_device"] = "cuda:0"
    with pytest.raises(ValueError, match="CPU auxiliary"):
        verify_auxiliary_models(CampaignConfig.model_validate(payload))
