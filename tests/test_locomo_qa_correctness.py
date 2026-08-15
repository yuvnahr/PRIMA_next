from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from benchmarks.common import BenchmarkArtifactStore
from benchmarks.locomo.adapter import LoCoMoAdapter, parse_timestamp
from benchmarks.locomo.evaluate import (
    LoCoMoEvaluator,
    answer_token_coverage,
    bert_scores_batch,
    evidence_recall,
    exact_match_score,
    f1_score,
    normalize_answer,
    rouge_l_score,
)
from benchmarks.locomo.experiment import (
    DEFAULT_POLICIES,
    LOCOMO_PROFILES,
    IngestionPolicy,
    _classify_failure,
    run_locomo_experiment,
    run_paired_locomo_experiment,
    select_conversations,
)
from runtime.contracts import (
    EvidenceReference,
    ExecutionOutcome,
    ExecutionProfile,
    ExecutionStatus,
    PrimaRequest,
    PrimaResponse,
    RuntimeDiagnostics,
)


def fixture() -> list[dict]:
    return [{
        "sample_id": "conversation-1",
        "conversation": {
            "speaker_a": "A",
            "speaker_b": "B",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "Alpha happened."}],
            "session_2_date_time": "2023-05-09T14:00:00Z",
            "session_2": [{"speaker": "B", "dia_id": "D2:1", "text": "Beta followed."}],
        },
        "qa": [
            {"id": "q1", "question": "What happened?", "answer": "Alpha", "evidence": ["D1:1"], "category": 2},
            {"id": "q2", "question": "Unknown?", "answer": None, "evidence": [], "category": 5},
            {"id": "q3", "question": "What happened across sessions?", "answer": "Alpha and Beta", "evidence": "D1:1; D2:1", "category": 1},
        ],
    }]


def write_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(fixture()), encoding="utf-8")
    return path


def make_response(request: PrimaRequest, *, failed: bool = False) -> PrimaResponse:
    ingestion = request.profile is ExecutionProfile.INGESTION_ONLY
    conversation = request.task_kind.value == "conversation"
    evidence = ()
    output_data = {}
    text = None
    if ingestion:
        output_data = {"memory_id": f"memory-{request.metadata['document_metadata']['source_turn_id']}"}
    elif conversation:
        text = "acknowledged"
        output_data = {"memory_ids_created": [f"memory-{request.metadata['turn_id']}"], "memory_admission": {"stored": True}}
    else:
        text = "Alpha"
        if request.profile is not ExecutionProfile.MODEL_ONLY:
            evidence = (EvidenceReference(
                source_id="memory-D1:1", text="Alpha happened.", score=1.0,
                metadata={"provenance": {"source_turn_id": "D1:1"}},
            ),)
            output_data = {"generation": {"selected_source_ids": ["memory-D1:1"]}, "retrieval": {"stage_source_ids": {
                "dense_top30": ["D1:1", "D2:1"], "sparse_top30": [],
                "fused_top30": ["D1:1", "D2:1"],
                "reranked_top30": ["D1:1"], "final_candidates": ["D1:1"],
            }}}
    return PrimaResponse(
        request_id=request.request_id, task_kind=request.task_kind, profile=request.profile,
        status=ExecutionStatus.FAILED if failed else ExecutionStatus.COMPLETED,
        outcome=ExecutionOutcome.FAILED if failed else (
            ExecutionOutcome.INGESTED if ingestion else ExecutionOutcome.ANSWERED
        ),
        output_text=text, output_data=output_data, evidence=evidence,
        diagnostics=RuntimeDiagnostics(
            route_name=f"{request.task_kind.value}:{request.profile.value}",
            latency_ms=3.0, retrieval_count=len(evidence),
            reflection_count=1 if request.profile is ExecutionProfile.PRIMA_FULL else 0,
            model_call_count=0 if ingestion else 1,
            model_usage={"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            maintenance={"enabled": False, "queue_size": 0, "failure_count": 0},
        ),
        errors=("synthetic failure",) if failed else (),
    )


class FakeRuntime:
    requests: list[PrimaRequest] = []
    client_count = 0

    def __init__(self, llm_client=None, **_config):
        if llm_client is None:
            llm_client = object()
            type(self).client_count += 1
        self.llm_client = llm_client

    async def execute(self, request: PrimaRequest) -> PrimaResponse:
        type(self).requests.append(request)
        return make_response(request)


class ExplodingRuntime(FakeRuntime):
    async def execute(self, request: PrimaRequest) -> PrimaResponse:
        if request.task_kind.value == "factual_qa" and "Unknown?" in request.input_text:
            raise RuntimeError("interrupted")
        return await super().execute(request)


def reset_fake() -> None:
    FakeRuntime.requests, FakeRuntime.client_count = [], 0


def test_documented_normalization_exact_match_and_multiplicity() -> None:
    assert normalize_answer("The Friday!") == "friday"
    assert exact_match_score("The Friday!", "friday") == 1.0
    assert exact_match_score("Alice and Bob", "Alice Bob") == 0.0
    assert exact_match_score("Bob Alice", "Alice Bob") == 0.0
    assert f1_score("cats cats", "cats") == pytest.approx(2 / 3)
    assert rouge_l_score("finished screenplay", "she finished screenplay") == 0.8
    assert evidence_recall(["D1:1", "D2:1"], ["D2:1"]) == 0.5
    assert answer_token_coverage("Alpha Beta", ["Beta followed."]) == 0.5


def test_timestamp_schema_category_and_evidence_validation() -> None:
    conversations = LoCoMoAdapter().adapt(fixture())
    assert conversations[0].turns[0].timestamp == "2023-05-08T13:56:00+00:00"
    assert conversations[0].questions[1].answer == "No information available"
    assert conversations[0].questions[2].evidence == ("D1:1", "D2:1")
    assert parse_timestamp("2023-05-08").isoformat() == "2023-05-08T00:00:00+00:00"
    assert parse_timestamp("08 May 2023 01:56 PM").isoformat() == "2023-05-08T13:56:00+00:00"
    scalar = fixture()
    scalar[0]["qa"][0]["answer"] = 2022
    assert LoCoMoAdapter().adapt(scalar)[0].questions[0].answer == "2022"
    invalid = fixture()
    invalid[0]["qa"][0]["category"] = 9
    with pytest.raises(ValueError, match="category must be"):
        LoCoMoAdapter().adapt(invalid)
    invalid = fixture()
    invalid[0]["qa"][0]["evidence"] = ["D"]
    with pytest.raises(ValueError, match="question 0.*malformed evidence"):
        LoCoMoAdapter().adapt(invalid)
    invalid = fixture()
    invalid[0]["conversation"]["session_1_date_time"] = "yesterday-ish"
    with pytest.raises(ValueError, match="conversation-1 session_1.*unsupported timestamp"):
        LoCoMoAdapter().adapt(invalid)
    invalid = fixture()
    invalid[0]["conversation"]["session_1"][0]["text"] = ""
    with pytest.raises(ValueError, match="session_1 turn 0.*text must be"):
        LoCoMoAdapter().adapt(invalid)
    invalid = fixture()
    invalid[0]["conversation"]["session_1"] = "not-a-list"
    with pytest.raises(ValueError, match="session_1.*non-empty list"):
        LoCoMoAdapter().adapt(invalid)
    invalid = fixture()
    del invalid[0]["conversation"]["speaker_a"]
    with pytest.raises(ValueError, match="speaker_a"):
        LoCoMoAdapter().adapt(invalid)
    invalid = fixture()
    invalid[0]["qa"][0]["evidence"] = ["D9:9"]
    with pytest.raises(ValueError, match="evidence IDs not present"):
        LoCoMoAdapter().adapt(invalid)


def test_optional_metrics_are_opt_in_and_bertscore_uses_device_and_batch(monkeypatch) -> None:
    records = [{"prediction": "alpha", "expected_answer": "alpha", "category": "3", "execution_failed": False}]
    core = LoCoMoEvaluator().evaluate(records)
    assert core["rouge_l"] is None and core["rouge_l_status"] == "disabled"
    assert core["bertscore"] is None and core["bertscore_status"] == "disabled"
    assert LoCoMoEvaluator(include_rouge_l=True).evaluate(records)["rouge_l"] == 1.0
    unavailable = LoCoMoEvaluator(include_bertscore=True).evaluate(records)
    assert str(unavailable["bertscore_status"]).startswith(("unavailable", "available"))
    calls = {}
    monkeypatch.setattr("benchmarks.locomo.evaluate.missing_modules", lambda _modules: ())
    monkeypatch.setitem(sys.modules, "bert_score", types.SimpleNamespace(
        score=lambda predictions, references, **kwargs: (
            calls.update(kwargs) or [0.0], [0.0], [0.75],
        ),
    ))
    assert bert_scores_batch(["a"], ["a"], device="cuda:1", batch_size=7) == ([0.75], "available")
    assert calls["device"] == "cuda:1" and calls["batch_size"] == 7


@pytest.mark.parametrize(
    ("updates", "profile", "expected"),
    [
        ({"ingestion_error": "bad"}, ExecutionProfile.PRIMA_FULL, "INGESTION_FAILURE"),
        ({"runtime_error": "bad"}, ExecutionProfile.PRIMA_FULL, "RUNTIME_FAILURE"),
        ({"prediction": ""}, ExecutionProfile.PRIMA_FULL, "ANSWER_PARSE_FAILURE"),
        ({"candidate_evidence_recall": 0.0}, ExecutionProfile.PRIMA_FULL, "RETRIEVAL_MISS"),
        ({"candidate_evidence_recall": 1.0, "final_evidence_recall": 0.5}, ExecutionProfile.PRIMA_FULL, "EVIDENCE_SELECTION_FAILURE"),
        ({"prediction": "wrong"}, ExecutionProfile.MODEL_ONLY, "ANSWER_SCORING_FAILURE"),
    ],
)
def test_every_failure_category_has_a_reachable_branch(updates, profile, expected) -> None:
    record = {
        "ingestion_error": None, "runtime_error": None, "prediction": "alpha",
        "expected_answer": "alpha", "expected_evidence": ["D1:1"],
        "candidate_evidence_recall": 1.0, "final_evidence_recall": 1.0,
        "execution_failed": False,
    }
    record.update(updates)
    assert _classify_failure(record, profile) == expected


def test_category_five_typed_abstention_is_not_an_evaluation_failure() -> None:
    record = {
        "ingestion_error": None, "runtime_error": None, "prediction": "",
        "expected_answer": "No information available", "expected_evidence": [],
        "candidate_evidence_recall": 0.0, "final_evidence_recall": 0.0,
        "category": "5", "outcome": "abstained", "execution_failed": False,
    }
    assert _classify_failure(record, ExecutionProfile.PRIMA_FULL) is None


@pytest.mark.parametrize("profile", LOCOMO_PROFILES)
def test_profiles_use_execute_and_explicit_ingestion_policies(tmp_path: Path, profile: ExecutionProfile) -> None:
    reset_fake()
    result = run_locomo_experiment(
        dataset_path=str(write_fixture(tmp_path)), output_path=str(tmp_path / "out"),
        runtime_profile=profile, max_conversations=1, max_questions=1,
        runtime_factory=FakeRuntime, provider="test", model="fixture-model",
    )
    assert result["ingestion_policy"] == DEFAULT_POLICIES[profile].value
    assert all(isinstance(request, PrimaRequest) for request in FakeRuntime.requests)
    assert FakeRuntime.requests[-1].profile is profile
    if profile is ExecutionProfile.MODEL_ONLY:
        assert FakeRuntime.requests[-1].input_text.startswith("Bounded conversation context:")
        assert "[2023-05-08T13:56:00+00:00]" in FakeRuntime.requests[-1].input_text
    turn_profiles = [request.profile for request in FakeRuntime.requests[:-1]]
    expected_turn_profile = (
        ExecutionProfile.PRIMA_FULL
        if DEFAULT_POLICIES[profile] is IngestionPolicy.NORMAL_PRIMA
        else ExecutionProfile.INGESTION_ONLY
    )
    assert turn_profiles == [expected_turn_profile, expected_turn_profile]
    for request in FakeRuntime.requests[:-1]:
        policy = (
            request.metadata["ingestion_policy"]
            if request.task_kind.value == "conversation"
            else request.metadata["document_metadata"]["ingestion_policy"]
        )
        assert policy == DEFAULT_POLICIES[profile].value


def test_question_checkpoint_resume_and_exact_duplicate_prevention(tmp_path: Path) -> None:
    dataset, output = write_fixture(tmp_path), tmp_path / "out"
    with pytest.raises(RuntimeError, match="interrupted"):
        run_locomo_experiment(
            dataset_path=str(dataset), output_path=str(output), runtime_profile="model_only",
            max_conversations=1, runtime_factory=ExplodingRuntime,
            provider="test", model="fixture-model",
        )
    store = BenchmarkArtifactStore(output / "model_only")
    assert [row.case_id for row in store.checkpoints()] == ["conversation-1:q1"]
    resumed = run_locomo_experiment(
        dataset_path=str(dataset), output_path=str(output), runtime_profile="model_only",
        max_conversations=1, runtime_factory=FakeRuntime, resume=True,
        provider="test", model="fixture-model",
    )
    assert resumed["questions"] == 3 and len(store.checkpoints()) == 3
    with pytest.raises(ValueError, match="model"):
        run_locomo_experiment(
            dataset_path=str(dataset), output_path=str(output), runtime_profile="model_only",
            max_conversations=1, runtime_factory=FakeRuntime, resume=True, model="changed",
            provider="test",
        )


def test_paired_modes_reconcile_metrics_memory_maintenance_and_selection(tmp_path: Path) -> None:
    reset_fake()
    results = run_paired_locomo_experiment(
        dataset_path=str(write_fixture(tmp_path)), output_path=str(tmp_path / "paired"),
        max_conversations=1, runtime_factory=FakeRuntime, parallel_workers=2,
        provider="test", model="fixture-model",
    )
    manifests = [BenchmarkArtifactStore(Path(result["artifacts"]["manifest"]).parent).read_manifest() for result in results.values()]
    assert {manifest.selected_ids for manifest in manifests} == {
        ("conversation-1:q1", "conversation-1:q2", "conversation-1:q3"),
    }
    assert {manifest.runtime_profile for manifest in manifests} == {profile.value for profile in LOCOMO_PROFILES}
    assert FakeRuntime.client_count == 1
    full = results["prima_full"]
    metrics = full["metrics"]
    assert full["questions"] == full["completed"] + full["failed"] == 3
    assert sum(item["question_count"] for item in metrics["category_metrics"].values()) == 3
    assert metrics["candidate_evidence_recall"] == 1.0
    assert metrics["final_evidence_recall"] == 0.75
    assert metrics["memory_admission_recall"] == 1.0
    assert metrics["memory_growth"] == 2
    assert metrics["maintenance_completion_rate"] == 1.0
    assert metrics["multi_session_breakdown"]["multi_session"]["question_count"] == 1
    assert metrics["multi_session_breakdown"]["unannotated"]["question_count"] == 1
    assert metrics["failure_taxonomy"]["total_failures"] == (
        metrics["failed_questions"] + metrics["evaluation_failure_questions"]
    )


def test_full_dataset_requires_explicit_flag() -> None:
    conversations = LoCoMoAdapter().adapt(fixture())
    with pytest.raises(ValueError, match="full_dataset"):
        select_conversations(conversations, seed=13, max_conversations=0, max_questions=0, full_dataset=False)
    assert select_conversations(conversations, seed=13, max_conversations=0, max_questions=0, full_dataset=True) == conversations
