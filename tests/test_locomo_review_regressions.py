"""Goal 1 regressions use synthetic histories, fake providers and a frozen upstream scorer."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import runpy
import subprocess  # nosec B404
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.common import BenchmarkArtifactStore
from benchmarks.locomo.adapter import LoCoMoAdapter
from benchmarks.locomo.evaluate import LoCoMoEvaluator, f1_score, normalized_exact_match
from benchmarks.locomo.experiment import IngestionPolicy, _checkpoint, _classify_failure, _question_record, _records
from benchmarks.locomo.official_scoring import UPSTREAM_COMMIT, official_locomo_score, scorer_identity
from llm.generation_config import GenerationConfig, StructuredOutputMode
from llm.llm_types import LLMResponse
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_confidence import RetrievalConfidence
from memory.retrieval.retrieval_controller import RetrievalResponse
from memory.retrieval.retrieval_result import RetrievalResult
from reasoning.controller import ReasoningController
from reasoning.evidence_audit import STAGES
from reasoning.models import ReasoningMode, ReasoningRequest
from runtime.contracts import (
    ExecutionOptions,
    ExecutionOutcome,
    ExecutionProfile,
    ExecutionStatus,
    PrimaRequest,
    PrimaResponse,
    TaskKind,
)
from runtime.prima_runtime import PrimaRuntime, _request_evidence_audit
from uncertainty import (
    ConfidenceSignal,
    ConfidenceSource,
    DecisionType,
    OverallConfidence,
    UncertaintyBand,
    UncertaintyEstimator,
)
from workflow.answer_generation import AnswerGenerationController, GenerationOutcome
from workflow.execution_context import ExecutionContext
from workflow.prima_workflow import EvidenceAcquisitionController

FIXTURES = Path(__file__).parent / "fixtures" / "locomo_review"


def retrieval(*turns: tuple[str, str, str]) -> RetrievalResponse:
    results = tuple(RetrievalResult(MemoryNote.create(text, MemoryType.SEMANTIC, note_id=memory,
                                                   embedding=(0.0,), context={"source_turn_id": turn} if turn else {}), 0.9)
                    for memory, turn, text in turns)
    rows = [{"id": result.note.id, **result.note.context} for result in results]
    return RetrievalResponse(results, RetrievalConfidence(0.9, 0.1, 1.0, 0.9),
                             {**dict.fromkeys(STAGES.values(), rows), "stage_limits": {"candidate_pool": 30, "returned": 5}})


class Client:
    provider_name = "test"

    def __init__(self, raw=None):
        self.raw = raw or json.dumps({"answer": "green", "evidence": ["E1"], "insufficient_information": False})
        self.prompts = []

    def chat(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        return LLMResponse(self.raw)


def test_earlier_hop_gold_survives_last_hop_miss_without_full_trace():
    responses = iter((retrieval(("first", "D1:1", "Alpha identifies Bridge.")),
                      retrieval(("second", "D2:1", "Bridge has the linked fact C."))))
    answer = ReasoningController().answer(
        ReasoningRequest("What fact is linked to Alpha?", session_id="conv"),
        retrieve=lambda query: next(responses), synthesize=lambda question, evidence: ("C", False, (), {}),
    )
    audit = answer.answer_diagnostics["evidence_audit"]
    assert len(answer.trace_summary) == 1
    assert len(audit["hops"]) == 2
    assert audit["candidate_source_turn_ids"] == ["conv:D1:1", "conv:D2:1"]
    assert audit["hops"][-1]["stage_source_ids"]["fused"] == ["conv:D2:1"]
    assert audit["accumulated_source_turn_ids"] == ["conv:D1:1", "conv:D2:1"]
    assert audit["hops"][0]["stage_memory_ids"]["fused"] == ["first"]
    assert audit["hops"][-1]["stop_reason"] == answer.stop_reason
    assert "text" not in json.dumps(audit)


def test_compressed_supplied_and_cited_ids_match_actual_prompt():
    response = retrieval(("one", "D1:1", "Oak color is green."), ("two", "D2:1", "Oak leaves are broad."),
                         ("three", "D1:1", "A second memory of the same turn."), ("summary", "", "Summary without lineage."))
    context = ExecutionContext("What color is Oak?", metadata={"session_id": "conv", "reasoning_mode": "single_pass", "max_context_tokens": 4})
    acquisition = EvidenceAcquisitionController(ReasoningController(), SimpleNamespace(retrieve=lambda request: response))
    context.reasoning_result = asyncio.run(acquisition.execute(context)).answer_result
    client = Client()
    context.metadata.update(task_kind="factual_qa", generation_config=GenerationConfig(
        model="fake", provider="test", structured_output=StructuredOutputMode.JSON_SCHEMA))
    context.generation_result = asyncio.run(AnswerGenerationController(client).execute(context))
    audit = _request_evidence_audit(context)
    assert context.generation_result.outcome is GenerationOutcome.ANSWERED
    assert audit["accumulated_source_turn_ids"] == ["conv:D1:1", "conv:D2:1"]
    assert audit["supplied_source_turn_ids"] == audit["cited_source_turn_ids"] == ["conv:D1:1"]
    assert audit["untraceable_accumulated_memory_count"] == 1
    prompt_records = json.loads(client.prompts[0].split("<UNTRUSTED_RETRIEVED_MEMORY>\n")[1].split("\n</UNTRUSTED_RETRIEVED_MEMORY>")[0])
    assert prompt_records == [{"label": "E1", "source_id": "E1", "text": "Oak color is green."}]
    assert audit["supplied_memory_ids"] == ["one"]
    assert audit["hops"][0]["added_source_turn_ids"] == ["conv:D1:1", "conv:D2:1"]


class RetryEstimator(UncertaintyEstimator):
    def estimate_from_subsystem_outputs(self, **kwargs):
        return OverallConfidence(confidence=0.4, uncertainty=0.54, confidence_interval=(0.4, 0.4),
                                 uncertainty_band=UncertaintyBand.LOW, decision_probabilities=(),
                                 recommended_decision=DecisionType.CONTINUE,
                                 signals=(ConfidenceSignal(ConfidenceSource.RETRIEVAL, 0.2, 0.8),))


def test_canonical_outer_retry_retains_all_hops_and_last_call_compatibility(tmp_path, monkeypatch):
    runtime = PrimaRuntime(llm_client=Client(), memory_repository=InMemoryMemoryRepository(),
                           uncertainty_estimator=RetryEstimator(), log_path=tmp_path / "runtime.log")
    responses = iter((retrieval(("first", "D1:1", "Oak color is green.")),
                      retrieval(("second", "D2:1", "Oak color is green too."))))
    monkeypatch.setattr(runtime.retrieval_controller, "retrieve", lambda request: next(responses))
    result = asyncio.run(runtime.execute(PrimaRequest(task_kind=TaskKind.FACTUAL_QA, profile=ExecutionProfile.PRIMA_FULL,
                                                     input_text="What color is Oak?", session_id="conv",
                                                     options=ExecutionOptions(reasoning_mode="single_pass"))))
    assert result.status is ExecutionStatus.COMPLETED
    assert result.output_data["retrieval_retry_count"] == 1
    assert result.diagnostics.retrieval_call_count == result.diagnostics.retrieval_count == 2
    audit = result.output_data["evidence_audit"]
    assert audit["candidate_source_turn_ids"] == ["conv:D1:1", "conv:D2:1"]
    assert [hop["hop_index"] for hop in audit["hops"]] == [0, 1]
    assert [hop["attempt_index"] for hop in audit["hops"]] == [0, 1]
    assert audit["hops"][1]["attempt_kind"] == "workflow_retry"
    assert result.output_data["retrieval"]["stage_source_ids"]["final_candidates"] == ["D2:1"]
    assert result.output_data["retrieval"]["scope"] == "last_retrieval_call"
    assert audit["supplied_source_turn_ids"] == audit["cited_source_turn_ids"] == ["conv:D2:1"]
    restored = PrimaResponse.from_dict(result.to_dict())
    assert restored.output_data["evidence_audit"] == audit


def test_canonical_answer_can_cite_earlier_hop_when_last_hop_misses_gold(tmp_path, monkeypatch):
    runtime = PrimaRuntime(llm_client=Client(), memory_repository=InMemoryMemoryRepository(), log_path=tmp_path / "runtime.log")
    responses = iter((retrieval(("first", "D1:1", "Alpha identifies Bridge.")),
                      retrieval(("second", "D2:1", "Bridge has the linked fact C."))))
    monkeypatch.setattr(runtime.retrieval_controller, "retrieve", lambda request: next(responses))
    result = asyncio.run(runtime.execute(PrimaRequest(task_kind=TaskKind.FACTUAL_QA, profile=ExecutionProfile.SIMPLE_RAG,
                                                     input_text="What fact is linked to Alpha?", session_id="conv")))
    audit = result.output_data["evidence_audit"]
    assert len(audit["hops"]) == 2
    assert audit["candidate_source_turn_ids"] == audit["supplied_source_turn_ids"] == ["conv:D1:1", "conv:D2:1"]
    assert audit["cited_source_turn_ids"] == ["conv:D1:1"]
    assert result.output_data["retrieval"]["stage_source_ids"]["final_candidates"] == ["D2:1"]


def test_phase_exception_retry_preserves_the_response_received_before_failure(tmp_path, monkeypatch):
    runtime = PrimaRuntime(llm_client=Client(), memory_repository=InMemoryMemoryRepository(), log_path=tmp_path / "runtime.log")
    responses = iter((retrieval(("first", "D1:1", "Oak color is green.")),
                      retrieval(("second", "D2:1", "Oak color is green too."))))
    monkeypatch.setattr(runtime.retrieval_controller, "retrieve", lambda request: next(responses))
    original_integrate = runtime.reasoning_controller.evidence_integrator.integrate
    calls = []

    def integrate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("synthetic phase failure after retrieval")
        return original_integrate(*args, **kwargs)

    monkeypatch.setattr(runtime.reasoning_controller.evidence_integrator, "integrate", integrate)
    result = asyncio.run(runtime.execute(PrimaRequest(task_kind=TaskKind.FACTUAL_QA, profile=ExecutionProfile.SIMPLE_RAG,
                                                     input_text="What color is Oak?", session_id="conv")))
    audit = result.output_data["evidence_audit"]
    assert audit["candidate_source_turn_ids"] == ["conv:D1:1", "conv:D2:1"]
    assert audit["accumulated_source_turn_ids"] == audit["supplied_source_turn_ids"] == ["conv:D2:1"]
    assert [hop["attempt_index"] for hop in audit["hops"]] == [0, 1]
    assert audit["hops"][0]["stop_reason"] == "integration_error"
    assert result.diagnostics.retrieval_call_count == 2


def test_caller_metadata_cannot_fabricate_observed_evidence():
    context = ExecutionContext("Question?", metadata={"evidence_audit": {"hops": [{"fake": True}]},
                                                     "prompt_evidence": [{"memory_id": "fake", "source_turn_id": "fake:D1:1"}]})
    audit = _request_evidence_audit(context)
    assert audit["hops"] == audit["supplied_source_turn_ids"] == []


def question_record(*, prediction="green", gold=("D1:1",), outcome=ExecutionOutcome.ANSWERED,
                    candidate=("conv:D1:1",), supplied=("conv:D1:1",), category="2", generation=None):
    question = SimpleNamespace(question="What color is Oak?", answer="green", category=category,
                               evidence=gold, question_id="q", metadata={})
    response = PrimaResponse(request_id="r", task_kind=TaskKind.FACTUAL_QA, profile=ExecutionProfile.PRIMA_FULL,
                             status=ExecutionStatus.COMPLETED, outcome=outcome, output_text=prediction,
                             output_data={"evidence_audit": {"schema_version": "1.0", "candidate_source_turn_ids": list(candidate),
                                          "accumulated_source_turn_ids": list(candidate), "supplied_source_turn_ids": list(supplied),
                                          "cited_source_turn_ids": list(supplied)}, "generation": generation or {}})
    return _question_record(SimpleNamespace(id="conv"), question, ExecutionProfile.PRIMA_FULL,
                            IngestionPolicy.NORMAL_PRIMA, response, set(gold), set())


def test_correct_answer_annotation_miss_is_only_an_independent_flag():
    record = question_record(candidate=(), supplied=())
    assert record["answer_correct"] and record["failure_category"] is None
    assert record["evidence_flags"]["candidate_incomplete"] is True
    metrics = LoCoMoEvaluator().evaluate([record])
    assert metrics["exact_match"] == 1.0
    assert metrics["failure_taxonomy"]["total_failures"] == 0
    assert metrics["evidence_populations"]["answerable"]["flags_non_additive"]["candidate_incomplete"] == 1


def test_partial_evidence_is_not_full_coverage_and_unannotated_rows_do_not_lower_mean():
    partial = question_record(gold=("D1:1", "D2:1"), prediction="wrong")
    assert partial["candidate_evidence_recall"] == partial["supplied_evidence_recall"] == 0.5
    assert partial["failure_category"] == "ANSWER_SCORING_FAILURE"
    no_annotation = question_record(gold=())
    assert no_annotation["evidence_diagnostics_status"] == "not_applicable"
    assert no_annotation["candidate_evidence_recall"] is None
    metrics = LoCoMoEvaluator().evaluate([partial, no_annotation])
    assert metrics["candidate_evidence_recall"] == 0.5
    assert metrics["evidence_populations"]["answerable"]["observed_counts"]["candidate_evidence_recall"] == 1


def test_structured_abstention_raw_official_score_and_parse_failure_are_distinct():
    record = question_record(prediction="I do not have sufficient evidence.", outcome=ExecutionOutcome.ABSTAINED,
                             category="5", candidate=(), supplied=())
    metrics = LoCoMoEvaluator().evaluate([record])
    assert record["failure_category"] is None
    assert metrics["abstention_category_5_accuracy"] == 1.0
    assert metrics["official_adversarial_raw_text_score"] == 0.0
    assert metrics["candidate_evidence_recall"] is None
    assert metrics["evidence_populations"]["adversarial"]["annotated_questions"] == 1
    record["final_evidence_ids"] = ["D1:1"]
    assert LoCoMoEvaluator().evaluate([record])["average_cited_evidence"] == 0.0
    failed = question_record(prediction="No information available", outcome=ExecutionOutcome.ABSTAINED, category="5",
                             generation={"failure_kind": "parse", "errors": ["invalid JSON"]})
    assert failed["failure_category"] == "ANSWER_PARSE_FAILURE"
    assert LoCoMoEvaluator().evaluate([failed])["abstention_category_5_accuracy"] == 0.0


@pytest.mark.parametrize(("category", "prediction", "reference"), [
    (1, "red", "red, blue"), (1, "blue, red", "red, blue"), (2, "May 8 2023", "8 May 2023"),
    (2, "August 15", "15 August 2023"), (3, "Paris", "Paris; open domain alternative"),
    (4, "running and cats", "run cat"), (5, "Not mentioned in the conversation", "adversarial"),
    (5, "I do not have sufficient evidence", "adversarial"), (2, "", ""), (4, "can't", "cant"),
])
def test_official_core_equivalence_against_frozen_pinned_upstream(category, prediction, reference):
    upstream = runpy.run_path(str(FIXTURES / "upstream_scoring_core.py"))
    expected, _, _ = upstream["eval_question_answering"]([
        {"category": category, "prediction": prediction, "answer": reference, "evidence": []}])
    assert official_locomo_score(prediction, reference, category) == pytest.approx(float(expected[0]))


def test_frozen_scorer_provenance_and_source_equivalence():
    fixture_source = (FIXTURES / "upstream_scoring_core.py").read_text(encoding="utf-8")
    assert hashlib.sha256(fixture_source.encode()).hexdigest() == "fba8820873e0d0194e50d36c2a774e1020bed4134bc0170373c36b5fbecfd4e5"
    assert scorer_identity()["upstream_commit"] == UPSTREAM_COMMIT
    source_path = Path(__file__).parents[1] / "benchmarks/locomo/external/task_eval/evaluation.py"
    if source_path.exists():
        source = source_path.read_text(encoding="utf-8")
        assert hashlib.sha256(source.encode()).hexdigest() == "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd"
        names = {node.name for node in ast.parse(fixture_source).body if isinstance(node, ast.FunctionDef)}
        original = {node.name: ast.dump(node) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names}
        assert original == {node.name: ast.dump(node) for node in ast.parse(fixture_source).body if isinstance(node, ast.FunctionDef)}


def test_official_legacy_and_strict_em_remain_separate():
    assert normalized_exact_match("Bob Alice", "Alice Bob") == 0
    assert f1_score("Bob Alice", "Alice Bob") == 1
    assert official_locomo_score("red", "red, blue", 1) == 0.5
    assert official_locomo_score("Paris", "Paris; other metadata", 3) == 1
    assert f1_score("Paris", "Paris; other metadata") < 1
    assert normalized_exact_match("May 8 2023", "8 May 2023") == 0


def test_reference_metadata_survives_adapter():
    record = {"sample_id": "conv", "conversation": {"speaker_a": "A", "speaker_b": "B",
              "session_1_date_time": "2023-05-08", "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "Paris"}]},
              "qa": [{"question": "Where?", "answer": "Paris; other reference", "category": 3, "evidence": ["D1:1"], "adversarial_answer": "Rome"}]}
    question = LoCoMoAdapter().adapt([record])[0].questions[0]
    assert question.metadata["reference_metadata"]["answer"] == "Paris; other reference"
    assert question.metadata["reference_metadata"]["adversarial_answer"] == "Rome"


def test_old_and_new_consumer_fixtures_and_checkpoint_roundtrip(tmp_path):
    fixtures = json.loads((FIXTURES / "report_contract.json").read_text(encoding="utf-8"))
    for fixture in fixtures:
        metrics = LoCoMoEvaluator().evaluate([fixture["record"]])
        for key, expected in fixture["expected_metrics"].items():
            assert metrics[key] == expected
    record = question_record()
    store = BenchmarkArtifactStore(tmp_path)
    _checkpoint(store, record)
    assert _records(store)[0]["evidence_audit"] == record["evidence_audit"]


def test_cross_conversation_turns_never_collide():
    audits = []
    for conversation in ("first", "second"):
        answer = ReasoningController().answer(ReasoningRequest("What color is Oak?", session_id=conversation, mode=ReasoningMode.SINGLE_PASS),
            retrieve=lambda query: retrieval(("one", "D1:1", "Oak color is green.")),
            synthesize=lambda question, evidence: ("green", False, (), {}))
        audits.extend(answer.answer_diagnostics["evidence_audit"]["candidate_source_turn_ids"])
    assert len(set(audits)) == 2


def test_missing_stage_data_is_unavailable_and_unknown_citations_fail():
    response = retrieval(("one", "D1:1", "Oak color is green."))
    response.diagnostics.clear()
    answer = ReasoningController().answer(ReasoningRequest("What color is Oak?", mode=ReasoningMode.SINGLE_PASS),
        retrieve=lambda query: response, synthesize=lambda question, evidence: ("green", False, (), {}))
    assert answer.answer_diagnostics["evidence_audit"]["candidate_source_turn_ids"] is None
    with pytest.raises(ValueError, match="unknown evidence"):
        AnswerGenerationController(Client())._parse('{"answer":"green","evidence":["E9"],"insufficient_information":false}',
            "Question?", {"E1": "one"}, True, "test", "fake", GenerationConfig(model="fake", provider="test"), 0.0, {})


def test_core_evaluation_never_imports_bertscore():
    checked = subprocess.run(  # noqa: S603  # nosec B603
        [sys.executable, "-c", "import sys; from benchmarks.locomo.evaluate import LoCoMoEvaluator; LoCoMoEvaluator().evaluate([]); assert 'bert_score' not in sys.modules"],
        capture_output=True, text=True, timeout=30,
    )
    assert checked.returncode == 0, checked.stderr


def test_excluded_evidence_and_fallbacks_do_not_become_citations():
    response = retrieval(("fact", "D1:1", "Oak color is green."), ("excluded", "D2:1", "Fact-free summary."))
    response.results[1].note.context["evidence_eligible"] = False
    context = ExecutionContext("What color is Oak?", metadata={"session_id": "conv", "reasoning_mode": "single_pass"})
    acquisition = EvidenceAcquisitionController(ReasoningController(), SimpleNamespace(retrieve=lambda request: response))
    context.reasoning_result = asyncio.run(acquisition.execute(context)).answer_result
    context.metadata.update(task_kind="factual_qa", generation_config=GenerationConfig(
        model="fake", provider="test", structured_output=StructuredOutputMode.JSON_SCHEMA))
    context.generation_result = asyncio.run(AnswerGenerationController(Client("invalid JSON")).execute(context))
    assert context.generation_result.failure_kind == "parse"
    audit = _request_evidence_audit(context)
    assert audit["hops"][0]["excluded_memory_ids"] == ["excluded"]
    assert audit["accumulated_source_turn_ids"] == audit["supplied_source_turn_ids"] == ["conv:D1:1"]
    assert audit["cited_source_turn_ids"] == []


def test_parse_failure_with_fail_policy_remains_a_parse_failure():
    question = SimpleNamespace(question="What color?", answer="green", category="2", evidence=(), question_id="q", metadata={})
    response = PrimaResponse(request_id="r", task_kind=TaskKind.FACTUAL_QA, profile=ExecutionProfile.MODEL_ONLY,
                             status=ExecutionStatus.FAILED, outcome=ExecutionOutcome.FAILED, errors=("invalid JSON",),
                             output_data={"generation": {"failure_kind": "parse", "errors": ["invalid JSON"]}})
    record = _question_record(SimpleNamespace(id="conv"), question, ExecutionProfile.MODEL_ONLY,
                              IngestionPolicy.CONTROLLED_DOCUMENT, response, set(), set())
    assert record["execution_failed"] and record["failure_category"] == "ANSWER_PARSE_FAILURE"


def test_primary_diagnostic_precedence_and_correct_f1_is_not_em():
    record = question_record(prediction="wrong", candidate=(), supplied=())
    record["memory_admission_recall"] = 0.0
    assert _classify_failure(record, ExecutionProfile.PRIMA_FULL) == "ADMISSION_COVERAGE_GAP"
    record.update(ingestion_error="bad", runtime_error="also bad", parse_error="invalid JSON")
    assert _classify_failure(record, ExecutionProfile.PRIMA_FULL) == "INGESTION_FAILURE"
    reordered = {"prediction": "Bob Alice", "expected_answer": "Alice Bob", "runtime_profile": "model_only"}
    metrics = LoCoMoEvaluator().evaluate([reordered])
    assert metrics["f1"] == 1.0 and metrics["exact_match"] == 0.0
    assert metrics["failure_taxonomy"]["counts"] == {"ANSWER_SCORING_FAILURE": 1}
