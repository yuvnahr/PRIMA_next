"""Deterministic Phase 06 correction-loop tests."""

from __future__ import annotations

import asyncio
import json

from affect.affect_engine import DynamicAffectEngine
from llm.generation_config import GenerationConfig, StructuredOutputMode
from llm.llm_types import LLMResponse
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_confidence import RetrievalConfidence
from memory.retrieval.retrieval_controller import RetrievalController, RetrievalResponse
from memory.retrieval.retrieval_result import RetrievalResult
from reasoning.controller import ReasoningController
from reasoning.reflection_advisor import ReflectionAction, ReflectionAdvice, ReflectionEvent
from reflection.reasoning_reflection_adapter import ReasoningReflectionAdapter
from reflection.reflection_engine import ReflectionEngine
from runtime import ExecutionProfile, PrimaRequest, PrimaRuntime, TaskKind
from runtime.route_profiles import select_route
from state.state_manager import InMemoryStateManager
from uncertainty import DecisionType, OverallConfidence, UncertaintyBand, UncertaintyEstimator
from workflow.correction_loop import CorrectionBudget, CorrectionLoop
from workflow.execution_context import ExecutionContext
from workflow.prima_workflow import PrimaWorkflow


def _confidence(confidence: float, uncertainty: float) -> OverallConfidence:
    return OverallConfidence(
        confidence=confidence,
        uncertainty=uncertainty,
        confidence_interval=(confidence, confidence),
        uncertainty_band=UncertaintyBand.HIGH if uncertainty >= 0.55 else UncertaintyBand.LOW,
        decision_probabilities=(),
        recommended_decision=DecisionType.REFLECT if uncertainty >= 0.55 else DecisionType.CONTINUE,
        signals=(),
    )


class _SequentialEstimator(UncertaintyEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def estimate_from_subsystem_outputs(self, **_kwargs: object) -> OverallConfidence:
        self.calls += 1
        return _confidence(0.35, 0.65) if self.calls == 1 else _confidence(0.90, 0.10)


class _HighEstimator(UncertaintyEstimator):
    def estimate_from_subsystem_outputs(self, **_kwargs: object) -> OverallConfidence:
        return _confidence(0.90, 0.10)


class _QueryRetrievalController(RetrievalController):
    def __init__(self, repository: InMemoryMemoryRepository) -> None:
        super().__init__(repository)
        self.queries: list[str] = []

    def retrieve(self, request):
        query = str(request.query)
        self.queries.append(query)
        if query == "corrected entity query":
            note = MemoryNote.create(
                "The corrected entity answer is Right.",
                MemoryType.SEMANTIC,
                note_id="correct",
                embedding=(0.0,),
            )
            confidence = RetrievalConfidence(0.9, 0.1, 1.0, 0.9)
        else:
            note = MemoryNote.create(
                "A distractor mentions an unrelated Wrong answer.",
                MemoryType.SEMANTIC,
                note_id="distractor",
                embedding=(0.0,),
            )
            confidence = RetrievalConfidence(0.2, 0.8, 0.2, 0.2)
        return RetrievalResponse((RetrievalResult(note, confidence.confidence),), confidence)


class _QueryAdvisor:
    def advise(self, event, _state, **_kwargs):
        return ReflectionAdvice(
            trigger=True,
            action=ReflectionAction.REVISE_QUERY,
            suggested_query="corrected entity query",
            correction_proposal="Retrieve independent evidence for the unresolved entity.",
            confidence=0.9,
            provenance={"component": "synthetic_reflection_advisor"},
            reason_code=event.value,
            event=event,
        )


class _FactualClient:
    provider_name = "test"

    def chat(self, **_kwargs: object) -> LLMResponse:
        return LLMResponse(json.dumps({"answer": "Right", "evidence": ["E1"], "insufficient_information": False}))


class _RegeneratingClient:
    provider_name = "test"

    def __init__(self, always_invalid: bool = False) -> None:
        self.calls = 0
        self.always_invalid = always_invalid

    def chat(self, **_kwargs: object) -> LLMResponse:
        self.calls += 1
        if self.always_invalid or self.calls == 1:
            return LLMResponse("")
        return LLMResponse("Corrected answer")


def test_pre_execution_reflection_changes_query_evidence_and_answer() -> None:
    repository = InMemoryMemoryRepository()
    retrieval = _QueryRetrievalController(repository)
    advisor = _QueryAdvisor()
    workflow = PrimaWorkflow.from_controllers(
        affect_engine=DynamicAffectEngine(),
        retrieval_controller=retrieval,
        reflection_engine=ReflectionEngine(trigger_threshold=0.0),
        reflection_advisor=advisor,
        reasoning_controller=ReasoningController(reflection_advisor=advisor),
        uncertainty_estimator=_SequentialEstimator(),
        llm_client=_FactualClient(),
        memory_repository=repository,
        state_manager=InMemoryStateManager(),
    )
    context = ExecutionContext(
        user_input="What is the entity answer?",
        metadata={
            "task_kind": "factual_qa",
            "profile": "prima_full",
            "reasoning_mode": "single_pass",
            "state_session_id": "phase06",
            "route": select_route(TaskKind.FACTUAL_QA, ExecutionProfile.PRIMA_FULL).phases,
            "generation_config": GenerationConfig(
                model="fixture",
                provider="test",
                structured_output=StructuredOutputMode.JSON_SCHEMA,
            ),
        },
    )

    result = asyncio.run(workflow.run(context.user_input, context))

    assert retrieval.queries == ["What is the entity answer?", "corrected entity query"]
    assert result.output["text"] == "Right"
    assert result.reasoning_result.evidence_references[0].source_id == "correct"
    attempt = result.correction_attempts[0]
    assert attempt.accepted
    assert attempt.before_query == "What is the entity answer?"
    assert attempt.after_query == "corrected entity query"
    assert attempt.before_answer == ""
    assert attempt.after_answer == "Right"
    assert "gold" not in json.dumps(attempt.to_dict()).lower()


def test_post_validation_regenerates_and_records_real_answers(tmp_path) -> None:
    client = _RegeneratingClient()
    runtime = PrimaRuntime(
        llm_client=client,
        memory_repository=InMemoryMemoryRepository(),
        log_path=tmp_path / "runtime.log",
    )
    response = asyncio.run(
        runtime.execute(
            PrimaRequest(
                task_kind=TaskKind.CONVERSATION,
                profile=ExecutionProfile.MODEL_ONLY,
                input_text="Give a response.",
            )
        )
    )

    assert client.calls == 2
    assert response.output_text == "Corrected answer"
    attempt = response.output_data["correction_attempts"][0]
    assert attempt["advice"]["action"] == "regenerate"
    assert attempt["before_answer"] == ""
    assert attempt["after_answer"] == "Corrected answer"
    assert response.output_data["correction_count"] == 1


def test_invalid_duplicate_and_budgeted_advice_are_rejected() -> None:
    context = ExecutionContext(user_input="Original query")
    loop = CorrectionLoop(CorrectionBudget(max_retrieval_retries=1, max_reflection_interventions=3))
    leaked = ReflectionAdvice(
        trigger=True,
        action=ReflectionAction.REVISE_QUERY,
        suggested_query="Use the gold answer",
        confidence=1.0,
        provenance={"source": "evaluation result"},
        reason_code="test",
    )
    valid = ReflectionAdvice(
        trigger=True,
        action=ReflectionAction.REVISE_QUERY,
        suggested_query="New query",
        confidence=0.9,
        provenance={"component": "reflection"},
        reason_code="low_confidence",
        event=ReflectionEvent.LOW_CONFIDENCE,
    )
    another = ReflectionAdvice(
        trigger=True,
        action=ReflectionAction.REVISE_QUERY,
        suggested_query="Another query",
        confidence=0.9,
        provenance={"component": "reflection"},
        reason_code="low_confidence",
    )

    assert loop.review(context, leaked, "pre_execution").decision_reason == "evaluation_leakage"
    assert loop.review(context, valid, "pre_execution").accepted
    assert loop.review(context, valid, "pre_execution").decision_reason == "duplicate_advice"
    assert loop.review(context, another, "pre_execution").decision_reason == "retrieval_budget_exhausted"

    replan_context = ExecutionContext(user_input="Plan")
    replan_loop = CorrectionLoop(CorrectionBudget(max_replans=1, max_reflection_interventions=3))
    replan = ReflectionAdvice(True, ReflectionAction.REPLAN, confidence=0.9, reason_code="constraint")
    second_replan = ReflectionAdvice(
        True,
        ReflectionAction.REPLAN,
        suggested_query="different signature",
        confidence=0.9,
        reason_code="constraint",
    )
    assert replan_loop.review(replan_context, replan, "pre_execution").accepted
    assert (
        replan_loop.review(replan_context, second_replan, "pre_execution").decision_reason
        == "replan_budget_exhausted"
    )


def test_reasoning_adapter_emits_usable_query_revision() -> None:
    advice = ReasoningReflectionAdapter(ReflectionEngine(trigger_threshold=0.0)).advise(
        ReflectionEvent.NO_NEW_EVIDENCE,
        object(),
        query="Original query",
        stop_reason="no_new_evidence",
    )

    assert advice.trigger
    assert advice.action is ReflectionAction.BROADEN_QUERY
    assert advice.suggested_query != "Original query"
    assert not advice.contains_evaluation_leakage()


def test_generation_budget_stops_repeated_schema_failures(tmp_path) -> None:
    client = _RegeneratingClient(always_invalid=True)
    runtime = PrimaRuntime(
        llm_client=client,
        memory_repository=InMemoryMemoryRepository(),
        correction_budget=CorrectionBudget(max_generations=2, max_reflection_interventions=3),
        log_path=tmp_path / "runtime.log",
    )

    response = asyncio.run(
        runtime.execute(
            PrimaRequest(
                task_kind=TaskKind.CONVERSATION,
                profile=ExecutionProfile.MODEL_ONLY,
                input_text="Give a response.",
            )
        )
    )

    assert client.calls == 2
    assert [item["decision_reason"] for item in response.output_data["correction_attempts"]] == [
        "accepted",
        "duplicate_advice",
    ]


def test_runtime_result_uses_actual_pre_and_post_correction_snapshots(tmp_path) -> None:
    client = _RegeneratingClient()
    runtime = PrimaRuntime(
        llm_client=client,
        memory_repository=InMemoryMemoryRepository(),
        uncertainty_estimator=_HighEstimator(),
        log_path=tmp_path / "runtime.log",
    )

    result = runtime.process("Give a response.")

    assert result.final_response == "Corrected answer"
    assert result.prediction_before_reflection == ""
    assert result.prediction_after_reflection == "Corrected answer"
    assert result.correction_count == 1
