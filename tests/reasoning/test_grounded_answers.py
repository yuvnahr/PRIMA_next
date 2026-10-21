"""Goal 3 route and provider-boundary checks without model/network access."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.common import BenchmarkArtifactStore
from benchmarks.locomo.experiment import IngestionPolicy, _checkpoint, _question_record, _records
from llm.generation_config import GenerationConfig, StructuredOutputMode
from llm.llm_client import LLMClient
from llm.llm_types import LLMResponse
from llm.provider import OpenAICompatibleProvider, ProviderError
from llm.request_budget import RequestBudget
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.retrieval.retrieval_confidence import RetrievalConfidence
from memory.retrieval.retrieval_controller import RetrievalResponse
from memory.retrieval.retrieval_result import RetrievalResult
from reasoning.controller import ReasoningController
from reasoning.information_need_generator import InformationNeedGenerator, normalize_query
from reasoning.models import (
    AnswerResult,
    EvidenceItem,
    EvidenceState,
    ReasoningBudget,
    ReasoningMode,
    ReasoningRequest,
    SufficiencyDecision,
    SufficiencyStatus,
)
from reasoning.reflection_advisor import ReflectionAction, ReflectionAdvice
from runtime.contracts import (
    ExecutionOptions,
    ExecutionOutcome,
    ExecutionProfile,
    ExecutionStatus,
    PrimaRequest,
    PrimaResponse,
    TaskKind,
)
from runtime.prima_runtime import PrimaRuntime
from workflow.answer_generation import AnswerGenerationController, GenerationOutcome
from workflow.correction_loop import CorrectionLoop
from workflow.execution_context import ExecutionContext
from workflow.prima_workflow import ReflectionController
from workflow.workflow_state import WorkflowPhase

QUESTION = "What day did Maya start volunteering?"
ANSWER = '{"answer":"15 August 2023","evidence":["E1"],"insufficient_information":false}'
CONFIG = GenerationConfig(model="fake", provider="test", structured_output=StructuredOutputMode.JSON_SCHEMA)


def response(text="on the 15th", *, eligible=True):
    note = MemoryNote.create(text, note_id="one", embedding=(1.0, 0.0), context={
        "source_timestamp": "2023-08-20T12:00:00+00:00", "source_turn_id": "D1:1", "source_session_id": "session_1",
        "source_conversation_id": "conv", "evidence_eligible": eligible})
    return RetrievalResponse((RetrievalResult(note, 0.9),), RetrievalConfidence(0.9, 0.1, 1.0, 0.9))


class Client:
    provider_name = "test"

    def __init__(self, text=ANSWER):
        self.text, self.calls = text, []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(self.text)


def runtime_for(tmp_path, monkeypatch, retrieved, client):
    runtime = PrimaRuntime(llm_client=client, memory_repository=InMemoryMemoryRepository(),
                           maintenance_enabled=False, log_path=tmp_path / "runtime.log")
    monkeypatch.setattr(runtime.retrieval_controller, "retrieve", lambda request: retrieved)
    return runtime


def context_for(*, status=SufficiencyStatus.NEED_MORE_EVIDENCE, errors=(), evidence=True, budget=None):
    items = (EvidenceItem("one:0", "Source timestamp: 2023-08-20\non the 15th", "one", "episodic", 0.9, 0,
                          QUESTION, {"source_turn_id": "D1:1"}),) if evidence else ()
    reasoning = AnswerResult("", status, 0.1, items, 1, "max_retrieval_calls", ReasoningBudget(),
                             errors=errors, answer_attempt_permitted=True)
    return ExecutionContext(QUESTION, reasoning_result=reasoning, request_budget=budget,
                            metadata={"task_kind": "factual_qa", "generation_config": CONFIG})


@pytest.mark.parametrize("route", ["default", "single_pass", "simple_rag", "full"])
def test_low_overlap_anchored_date_reaches_exactly_one_grounded_call(tmp_path, monkeypatch, route):
    client = Client()
    runtime = runtime_for(tmp_path, monkeypatch, response(), client)
    if route == "default":
        result = runtime.answer_question(QUESTION, session_id="conv")
        data = result.answer_diagnostics
        assert result.answer == "15 August 2023" and result.status is SufficiencyStatus.NEED_MORE_EVIDENCE
    else:
        result = runtime.execute_sync(PrimaRequest(task_kind=TaskKind.FACTUAL_QA,
            profile=ExecutionProfile.SIMPLE_RAG if route == "simple_rag" else ExecutionProfile.PRIMA_FULL,
            input_text=QUESTION, session_id="conv", generation_config=CONFIG,
            options=ExecutionOptions(reasoning_mode="single_pass" if route == "single_pass" else "adaptive")))
        data = result.output_data
        assert result.outcome is ExecutionOutcome.ANSWERED and result.output_text == "15 August 2023"
        assert data["sufficiency_status"] == "need_more_evidence"
    assert len(client.calls) == data["request_budget"]["llm_calls"] == 1
    assert data["answer_attempt_permitted"] is True
    assert data["stop_reason"] in {"single_pass", "no_supported_expansion"}
    assert data["answer_attempt"]["retrieval_stop_reason"] == data["stop_reason"]
    prompt = client.calls[0]["prompt"]
    assert "2023-08-20" in prompt and "on the 15th" in prompt
    assert data["evidence_audit"]["supplied_source_turn_ids"] == ["conv:D1:1"]
    assert data["evidence_audit"]["cited_source_turn_ids"] == ["conv:D1:1"]


@pytest.mark.parametrize("cause", ["empty", "missing_reasoning", "ineligible", "contradiction", "hard_error", "llm_budget", "deadline", "context"])
def test_workflow_gate_blocks_speculation_before_call(cause):
    client = Client()
    context = context_for()
    if cause == "empty":
        context.reasoning_result = replace(context.reasoning_result, evidence_references=())
    elif cause == "missing_reasoning":
        context.reasoning_result = None
    elif cause == "ineligible":
        item = replace(context.reasoning_result.evidence_references[0], provenance={"evidence_eligible": False})
        context.reasoning_result = replace(context.reasoning_result, evidence_references=(item,))
    elif cause == "contradiction":
        context.reasoning_result = replace(context.reasoning_result, status=SufficiencyStatus.CONTRADICTORY)
    elif cause == "hard_error":
        context.hard_error = "unresolved controller error"
    elif cause == "llm_budget":
        context.request_budget = RequestBudget(max_llm_calls=0)
    elif cause == "deadline":
        context.request_budget = RequestBudget(started_at=-1000)
    elif cause == "context":
        context.reasoning_result = replace(context.reasoning_result, effective_budget=ReasoningBudget(max_context_tokens=1))
    result = asyncio.run(AnswerGenerationController(client).execute(context))
    assert result.outcome is GenerationOutcome.ABSTAINED and client.calls == []
    assert result.blocked_reason


@pytest.mark.parametrize("status,evidence,calls", [
    (SufficiencyStatus.NEED_MORE_EVIDENCE, True, 0), (SufficiencyStatus.CONTRADICTORY, True, 4),
    (SufficiencyStatus.ERROR, True, 4), (SufficiencyStatus.NEED_MORE_EVIDENCE, False, 4)])
def test_reasoning_gate_is_independently_guarded(status, evidence, calls):
    invoked = []
    controller = ReasoningController(verifier=SimpleNamespace(evaluate=lambda state: SufficiencyDecision(status, 0.1, False)))
    result = controller.answer(ReasoningRequest(QUESTION, mode=ReasoningMode.SINGLE_PASS,
        budget=ReasoningBudget(max_llm_calls=calls)), retrieve=lambda query: response() if evidence else RetrievalResponse((), RetrievalConfidence(0, 1, 0, 0)),
        synthesize=lambda *args: invoked.append(args) or ("unsafe", True, (), {}))
    assert invoked == [] and not result.answer_attempt_permitted


@pytest.mark.parametrize("text,outcome", [
    ('{"answer":null,"evidence":[],"insufficient_information":true}', GenerationOutcome.ABSTAINED),
    ('{"answer":"15 August [E999]","evidence":["E1"],"insufficient_information":false}', GenerationOutcome.FAILED),
    ('{"answer":"15 August","evidence":["E999"],"insufficient_information":false}', GenerationOutcome.FAILED)])
def test_bounded_attempt_preserves_model_abstention_and_rejects_fabricated_citations(text, outcome):
    client = Client(text)
    context = context_for()
    generator = AnswerGenerationController(client)
    first = asyncio.run(generator.execute(context))
    context.generation_result = first
    second = asyncio.run(generator.execute(context))
    assert first.outcome is outcome and second is first and len(client.calls) == 1
    advice = ReflectionAdvice(True, ReflectionAction.REGENERATE, confidence=0.9)
    assert CorrectionLoop().review(context, advice, "post_execution").decision_reason == "answer_attempt_exhausted"


@pytest.mark.parametrize("candidate", ["Wow", " Your ", "IT", "Thanks", "Sounds", "  sounds, ", "TH-ANKS", "W_OW"])
def test_generic_source_words_do_not_become_followups(candidate):
    state = EvidenceState(ReasoningRequest("What did Maya do last year?"))
    state.evidence_items = [EvidenceItem("1", f"Maya knows {candidate}.", "1", "episodic", 0.9, 0, "")]
    decision = SufficiencyDecision(SufficiencyStatus.NEED_MORE_EVIDENCE, 0.1, False,
                                   missing_information=("Missing requested activity",), unresolved_entities=(candidate,))
    assert InformationNeedGenerator().generate(state, decision) is None


def test_expansion_uses_explicit_unresolved_fields_and_preserves_question_constraints():
    question = "Which hospital treated Maya's friend last year?"
    state = EvidenceState(ReasoningRequest(question))
    state.evidence_items = [EvidenceItem("1", "Maya's friend was called Cedar.", "1", "episodic", 0.9, 0, "")]
    decision = SufficiencyDecision(SufficiencyStatus.NEED_MORE_EVIDENCE, 0.1, False,
        missing_information=("Hospital treating the named friend",), unresolved_entities=("Cedar", "Invented Person"))
    generator = InformationNeedGenerator()
    need = generator.generate(state, decision)
    assert need and need.query == question + " Source bridge: Cedar"
    state.attempted_queries = [need.query.upper().replace(" ", "  ")]
    assert generator.generate(state, decision) is None
    assert generator.generate(state, replace(decision, unresolved_entities=(), missing_information=())) is None
    assert normalize_query(need.query).startswith(normalize_query(question))


def test_answer_reserve_stops_retrieval_before_deadline(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("llm.request_budget.monotonic", lambda: clock[0])
    shared = RequestBudget(time_budget_seconds=10, answer_reserve_seconds=2, started_at=100)
    retrieved, synthesized = [], []
    def retrieve(query):
        retrieved.append(query)
        clock[0] = 108.5
        return response("Maya identifies Cedar.")
    result = ReasoningController().answer(ReasoningRequest(QUESTION, shared_budget=shared), retrieve=retrieve,
        synthesize=lambda *args: synthesized.append(args) or ("grounded", True, (), {}))
    assert len(retrieved) == len(synthesized) == 1
    assert result.stop_reason == "answer_time_reserve" and result.answer_attempt_permitted
    assert shared.remaining_seconds == 1.5


def test_hard_deadline_suppresses_reasoning_and_workflow_reflection(monkeypatch):
    calls = []
    advisor = SimpleNamespace(advise=lambda *args, **kwargs: calls.append("advisor"))
    budget = RequestBudget(started_at=-1000)
    result = ReasoningController(reflection_advisor=advisor).answer(ReasoningRequest(QUESTION, shared_budget=budget),
        retrieve=lambda query: calls.append("retrieve"), synthesize=lambda *args: calls.append("synthesize"))
    assert result.stop_reason == "time_budget" and calls == []
    context = context_for(budget=budget)
    engine = SimpleNamespace(evaluate=lambda *args: calls.append("reflection"))
    assert asyncio.run(ReflectionController(engine, advisor).execute(context)) is None
    assert calls == []
    advice = ReflectionAdvice(True, ReflectionAction.REGENERATE, confidence=0.9)
    assert CorrectionLoop().review(context, advice, "post_execution").decision_reason == "time_budget"


class RetryingProvider:
    capabilities = OpenAICompatibleProvider.capabilities

    def __init__(self, failures):
        self.failures, self.requests = failures, []

    def validate(self, request):
        pass

    def send(self, request):
        self.requests.append(request)
        if len(self.requests) <= self.failures:
            raise ProviderError("temporary")
        return LLMResponse(ANSWER)


@pytest.mark.parametrize("failures,limit,expected_calls", [(10, 1, 1), (1, 2, 2)])
def test_provider_retries_use_same_pre_call_budget(failures, limit, expected_calls):
    provider = RetryingProvider(failures)
    client = LLMClient(provider_name="test", provider=provider)
    context = context_for(status=SufficiencyStatus.SUFFICIENT, budget=RequestBudget(max_llm_calls=limit))
    context.metadata["generation_config"] = CONFIG.with_overrides(retries=4)
    result = asyncio.run(AnswerGenerationController(client).execute(context))
    assert len(provider.requests) == context.request_budget.llm_calls == context.metadata["model_call_count"] == expected_calls
    assert all(request.generation.timeout_seconds <= 15 for request in provider.requests)
    assert result.outcome is (GenerationOutcome.ANSWERED if failures == 1 else GenerationOutcome.ABSTAINED)


def test_bounded_attempt_does_not_retry_provider_with_unspent_calls():
    provider = RetryingProvider(10)
    context = context_for(budget=RequestBudget(max_llm_calls=4))
    context.metadata["generation_config"] = CONFIG.with_overrides(retries=4)
    result = asyncio.run(AnswerGenerationController(LLMClient(provider_name="test", provider=provider)).execute(context))
    assert len(provider.requests) == context.request_budget.llm_calls == 1
    assert context.request_budget.bounded_answer_attempts == 1
    assert result.outcome is GenerationOutcome.FAILED


def test_old_answer_records_keep_compatibility_without_granting_low_overlap_permission():
    old = AnswerResult("", SufficiencyStatus.NEED_MORE_EVIDENCE, 0.1, (), 1, "max_hops", ReasoningBudget())
    assert old.answer_attempt_permitted is None
    assert old.to_dict()["answer_attempt_permitted"] is None


@pytest.mark.parametrize("permission,called", [(None, 1), (False, 0)])
def test_legacy_sufficient_status_is_compatible_but_explicit_denial_is_authoritative(permission, called):
    context = context_for(status=SufficiencyStatus.SUFFICIENT)
    context.reasoning_result = replace(context.reasoning_result, answer_attempt_permitted=permission)
    client = Client()
    asyncio.run(AnswerGenerationController(client).execute(context))
    assert len(client.calls) == called


@pytest.mark.parametrize("limit,expected", [(1, 1), (2, 2)])
def test_runtime_corrections_share_the_original_call_allowance(tmp_path, limit, expected):
    class CorrectingClient(Client):
        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return LLMResponse("" if len(self.calls) == 1 else "Corrected answer")

    client = CorrectingClient()
    runtime = PrimaRuntime(llm_client=client, memory_repository=InMemoryMemoryRepository(),
                           maintenance_enabled=False, log_path=tmp_path / "runtime.log")
    result = runtime.execute_sync(PrimaRequest(task_kind=TaskKind.CONVERSATION, profile=ExecutionProfile.MODEL_ONLY,
        input_text="Give a response.", options=ExecutionOptions(max_llm_calls=limit)))
    assert len(client.calls) == result.diagnostics.model_call_count == result.output_data["request_budget"]["llm_calls"] == expected
    if limit == 1:
        assert result.output_data["correction_attempts"][0]["decision_reason"] == "max_llm_calls"
    else:
        assert result.output_text == "Corrected answer"


def test_provider_retry_cannot_cross_the_deadline(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("llm.request_budget.monotonic", lambda: clock[0])
    class ExpiringProvider(RetryingProvider):
        def send(self, request):
            self.requests.append(request)
            clock[0] = 120.0
            raise ProviderError("request exhausted the deadline")

    provider = ExpiringProvider(1)
    context = context_for(status=SufficiencyStatus.SUFFICIENT,
                          budget=RequestBudget(time_budget_seconds=10, started_at=100))
    context.metadata["generation_config"] = CONFIG.with_overrides(retries=3)
    result = asyncio.run(AnswerGenerationController(LLMClient(provider_name="test", provider=provider)).execute(context))
    assert len(provider.requests) == 1 and result.blocked_reason == "time_budget"


def test_shared_retrieval_budget_survives_outer_workflow_retry(tmp_path, monkeypatch):
    runtime = runtime_for(tmp_path, monkeypatch, response(), Client())
    acquisition = runtime.workflow.engine.registry.get(WorkflowPhase.EVIDENCE_ACQUISITION)
    context = ExecutionContext(QUESTION, request_budget=RequestBudget(),
        metadata={"max_retrieval_calls": 1, "reasoning_mode": "single_pass", "session_id": "conv"})
    first = asyncio.run(acquisition.execute(context)).answer_result
    context.reasoning_result = first
    second = asyncio.run(acquisition.execute(context)).answer_result
    assert context.request_budget.retrieval_calls == 1
    assert first.answer_attempt_permitted and second.stop_reason == "max_retrieval_calls"


def test_retrieval_finishing_after_deadline_cannot_open_either_gate(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("llm.request_budget.monotonic", lambda: clock[0])
    budget = RequestBudget(started_at=100, time_budget_seconds=10)
    calls = []
    def retrieve(query):
        clock[0] = 111.0
        return response("Maya started volunteering on the 15th.")
    result = ReasoningController().answer(ReasoningRequest(QUESTION, shared_budget=budget), retrieve=retrieve,
        synthesize=lambda *args: calls.append(args))
    assert result.stop_reason == "time_budget" and not result.answer_attempt_permitted and not calls
    context = context_for(budget=budget)
    context.reasoning_result = result
    client = Client()
    assert asyncio.run(AnswerGenerationController(client).execute(context)).blocked_reason == "time_budget"
    assert not client.calls


@pytest.mark.parametrize("example_index", [0, 1])
def test_old_new_answer_diagnostics_export_and_resume_without_invented_observations(tmp_path, example_index):
    fixture_path = Path(__file__).parents[1] / "fixtures/locomo_review/goal3_answer_contract.json"
    example = json.loads(fixture_path.read_text(encoding="utf-8"))["examples"][example_index]
    data = example["runtime_output_data"]
    response_record = PrimaResponse(request_id="r", task_kind=TaskKind.FACTUAL_QA, profile=ExecutionProfile.SIMPLE_RAG,
        status=ExecutionStatus.COMPLETED, outcome=ExecutionOutcome.ANSWERED, output_text="15 August 2023", output_data=data)
    question = SimpleNamespace(question=QUESTION, answer="15 August 2023", category="2", evidence=(), question_id="q", metadata={})
    record = _question_record(SimpleNamespace(id="conv"), question, ExecutionProfile.SIMPLE_RAG,
        IngestionPolicy.CONTROLLED_DOCUMENT, response_record, set(), set())
    for key in ("answer_attempt_permitted", "answer_attempt", "request_budget"):
        assert record[key] == data.get(key)
    if example_index:
        assert data["sufficiency_status"] == "need_more_evidence" and data["answer_attempt_permitted"] is True
    store = BenchmarkArtifactStore(tmp_path)
    _checkpoint(store, record)
    restored = _records(store)[0]
    assert restored["request_budget"] == record["request_budget"]
