from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from benchmarks.common import (
    BenchmarkArtifactStore,
    BenchmarkLifecycle,
    BenchmarkManifest,
    BenchmarkMode,
    BenchmarkSpec,
    BenchmarkSummary,
    CheckpointRecord,
    ClassificationCase,
    ConversationQACase,
    FailureCategory,
    FailureRecord,
    ItemTiming,
    LifecycleStage,
    PredictionRecord,
    ProgressEvent,
    ResumeCompatibilityError,
    RunStatus,
    TokenUsage,
    atomic_write_json,
    resume_manifest,
    stable_case_id,
    validate_resume,
)
from benchmarks.common import artifacts as artifact_module
from benchmarks.goemotions.dataset import GoEmotionsExample
from benchmarks.hotpotqa.adapter import HotpotQAAdapter
from benchmarks.locomo.adapter import LoCoMoAdapter

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _manifest(**updates) -> BenchmarkManifest:
    values = {
        "campaign_id": "campaign-a",
        "benchmark": BenchmarkSpec(
            name="fixture",
            version="1",
            mode=BenchmarkMode.CONVERSATION_QA,
            dataset_name="fixture-data",
        ),
        "git_commit": "abc123",
        "dataset_hash": "dataset-sha256",
        "selected_ids": ("case-1",),
        "provider": "test",
        "model": "model-a",
        "model_revision": "rev-1",
        "generation_config": {"temperature": 0.0, "max_output_tokens": 32},
        "runtime_profile": "prima_full",
        "active_capabilities": {"retrieval": True},
        "repository_mode": "benchmark:in_memory",
        "prompt_hashes": {"answer": "prompt-sha256"},
        "seed": 13,
        "dependencies": {"pydantic": "2"},
        "python_version": "3.10.11",
        "hardware": {"machine": "test"},
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(updates)
    return BenchmarkManifest(**values)


def _prediction(case_id: str = "case-1") -> PredictionRecord:
    return PredictionRecord(
        case_id=case_id,
        mode=BenchmarkMode.CONVERSATION_QA,
        prediction="answer",
        timing=ItemTiming(started_at=NOW, finished_at=NOW, total_ms=4.5, provider_ms=2.0),
        tokens=TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )


def _complete_checkpoint(case_id: str = "case-1") -> CheckpointRecord:
    return CheckpointRecord(case_id=case_id, status=RunStatus.COMPLETE, prediction=_prediction(case_id))


def _summary(status: RunStatus = RunStatus.COMPLETE) -> BenchmarkSummary:
    return BenchmarkSummary(
        campaign_id="campaign-a",
        status=status,
        total_cases=1,
        completed_cases=1 if status is RunStatus.COMPLETE else 0,
        failed_cases=0,
        cancelled_cases=0,
        started_at=NOW,
        finished_at=NOW,
    )


def test_atomic_json_interruption_preserves_previous_file(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("simulated interruption")

    monkeypatch.setattr(artifact_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interruption"):
        atomic_write_json(destination, {"new": True})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_malformed_checkpoint_row_is_rejected(tmp_path) -> None:
    store = BenchmarkArtifactStore(tmp_path / "run")
    store.layout.checkpoints.parent.mkdir(parents=True)
    store.layout.checkpoints.write_text('{"case_id":"ok"}\n{"broken":', encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed checkpoint row.*:2"):
        store.checkpoints()


def test_stable_ids_and_duplicate_suppression(tmp_path) -> None:
    identity = {"conversation": "c", "question": 2}
    assert stable_case_id("LoCoMo", identity) == stable_case_id("LoCoMo", dict(reversed(identity.items())))
    with pytest.raises(ValueError, match="unique"):
        _manifest(selected_ids=("case-1", "case-1"))

    store = BenchmarkArtifactStore(tmp_path / "run")
    store.initialize(_manifest())
    assert store.append_checkpoint(_complete_checkpoint()) is True
    assert store.append_checkpoint(_complete_checkpoint()) is False
    assert len(store.checkpoints()) == 1


def test_resume_rejects_changed_model_and_dataset() -> None:
    existing = _manifest()
    with pytest.raises(ResumeCompatibilityError, match="model"):
        validate_resume(existing, _manifest(campaign_id="b", model="model-b"))
    with pytest.raises(ResumeCompatibilityError, match="dataset_hash"):
        validate_resume(existing, _manifest(campaign_id="b", dataset_hash="different"))


def test_exact_resume_records_lineage() -> None:
    existing = _manifest(status=RunStatus.FAILED)
    resumed = resume_manifest(existing, _manifest(campaign_id="campaign-b"))

    assert resumed.status is RunStatus.PARTIAL
    assert resumed.resume_lineage[0].campaign_id == "campaign-a"
    assert resumed.resume_lineage[0].manifest_fingerprint == existing.manifest_fingerprint


def test_partial_and_cancelled_campaigns_cannot_appear_complete(tmp_path) -> None:
    store = BenchmarkArtifactStore(tmp_path / "run")
    store.initialize(_manifest())

    with pytest.raises(ValueError, match="exact selected IDs"):
        store.finalize(_summary())
    assert store.read_manifest().status is RunStatus.PARTIAL

    cancelled = store.mark_status(RunStatus.CANCELLED)
    assert cancelled.status is RunStatus.CANCELLED
    assert store.read_manifest().status is RunStatus.CANCELLED


def test_atomic_finalization_writes_completion_marker_last(tmp_path, monkeypatch) -> None:
    store = BenchmarkArtifactStore(tmp_path / "run")
    store.initialize(_manifest())
    store.append_checkpoint(_complete_checkpoint())
    real_replace = artifact_module.os.replace

    def interrupt_manifest(source, destination):
        if destination == store.layout.manifest:
            raise OSError("final marker interrupted")
        real_replace(source, destination)

    monkeypatch.setattr(artifact_module.os, "replace", interrupt_manifest)
    with pytest.raises(OSError, match="final marker"):
        store.finalize(_summary())

    assert store.read_manifest().status is RunStatus.PARTIAL
    assert json.loads(store.layout.summary.read_text(encoding="utf-8"))["status"] == "complete"


def test_task_shapes_timing_tokens_failures_and_progress_are_independent() -> None:
    classification = ClassificationCase(case_id="emotion-1", text="happy", gold_labels=("joy",))
    qa = ConversationQACase(
        case_id="qa-1",
        conversation_id="conversation-1",
        turns=({"speaker": "A", "text": "hello"},),
        question="Who spoke?",
        gold_answer="A",
    )
    failure = FailureRecord(
        case_id="qa-1",
        category=FailureCategory.RUNTIME,
        subcode="HOTpot.RUNTIME_PROVIDER",
        stage=LifecycleStage.EXECUTE_CASE,
        message="offline",
    )
    checkpoint = CheckpointRecord(case_id="qa-1", status=RunStatus.FAILED, failure=failure)

    assert classification.gold_labels == ("joy",)
    assert qa.gold_answer == "A"
    assert checkpoint.failure and checkpoint.failure.subcode == "HOTpot.RUNTIME_PROVIDER"
    assert _prediction().timing.total_ms == 4.5
    assert _prediction().tokens.total_tokens == 7


def test_lifecycle_emits_shared_events_and_rejects_backward_transitions() -> None:
    class Sink:
        def __init__(self) -> None:
            self.events: list[ProgressEvent] = []

        def publish(self, event: ProgressEvent) -> None:
            self.events.append(event)

    sink = Sink()
    lifecycle = BenchmarkLifecycle(sink)
    lifecycle.transition(LifecycleStage.PREFLIGHT)
    lifecycle.transition(LifecycleStage.LOAD)
    lifecycle.transition(LifecycleStage.EXECUTE_CASE, current=1, total=2, case_id="case-1")

    assert [event.stage for event in sink.events] == [
        LifecycleStage.PREFLIGHT,
        LifecycleStage.LOAD,
        LifecycleStage.EXECUTE_CASE,
    ]
    with pytest.raises(ValueError, match="cannot move backward"):
        lifecycle.transition(LifecycleStage.SELECT_CASES)


def test_manifest_rejects_credentials_and_url_tokens() -> None:
    with pytest.raises(ValueError, match="credentials"):
        _manifest(generation_config={"openai_api_key": "raw-secret"})
    with pytest.raises(ValueError, match="URLs"):
        _manifest(active_capabilities={"endpoint": "https://example.test/run?api_key=raw-secret"})


def test_artifact_writes_redact_nested_credentials(tmp_path) -> None:
    store = BenchmarkArtifactStore(tmp_path / "run")
    prediction = _prediction().model_copy(
        update={
            "diagnostics": {
                "Authorization": "Bearer raw-secret",
                "endpoint": "https://example.test/run?token=raw-secret",
            }
        }
    )
    store.append_prediction(prediction)

    written = store.layout.predictions.read_text(encoding="utf-8")
    assert "raw-secret" not in written
    assert "REDACTED" in written


def test_all_three_benchmark_shapes_can_adopt_contract_without_gold_changes() -> None:
    emotion = GoEmotionsExample("joyful", frozenset({"joy", "approval"}), "emotion-1")
    classification = ClassificationCase(
        case_id=emotion.example_id,
        text=emotion.text,
        gold_labels=tuple(sorted(emotion.labels)),
    )
    hotpot = HotpotQAAdapter().adapt(
        [{"_id": "hotpot-1", "question": "Who?", "answer": "Ada", "context": [["Doc", ["Ada."]]]}]
    )[0]
    locomo = LoCoMoAdapter().adapt(
        [{"id": "locomo-1", "conversation": {}, "qa": [{"id": "q1", "question": "Who?", "answer": "Lin"}]}]
    )[0]
    qa_cases = [
        ConversationQACase(
            case_id=conversation.questions[0].question_id or conversation.id,
            conversation_id=conversation.id,
            turns=tuple({"speaker": turn.speaker, "text": turn.text} for turn in conversation.turns),
            question=conversation.questions[0].question,
            gold_answer=conversation.questions[0].answer,
        )
        for conversation in (hotpot, locomo)
    ]

    assert classification.gold_labels == ("approval", "joy")
    assert [case.gold_answer for case in qa_cases] == ["Ada", "Lin"]
