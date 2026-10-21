"""Final producer-contract invariants with synthetic sources and offline providers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.locomo.adapter import LoCoMoAdapter
from benchmarks.locomo.evaluate import paired_conversation_comparison, paired_conversation_interval
from benchmarks.locomo.experiment import IngestionPolicy, _run_conversation, _turn_request
from config.runtime_config import RuntimeConfig
from llm.generation_config import GenerationConfig
from llm.request_budget import RequestBudget
from memory.evolution.memory_evolution_engine import MemoryEvolutionEngine
from memory.evolution.semantic_abstraction_engine import SemanticAbstractionEngine, factual_evidence_eligible
from memory.graph.graph_repository import GraphRepository
from memory.maintenance.background_supervisor import MaintenanceMode
from memory.maintenance.importance_types import MemoryImportanceConfig
from memory.maintenance.memory_importance import MemoryImportanceEngine
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryLevel, MemoryType
from memory.retrieval.reranker import Reranker
from memory.retrieval.retrieval_controller import RetrievalController
from memory.retrieval.retrieval_request import RetrievalRequest
from reasoning.models import AnswerResult, EvidenceItem, ReasoningBudget, SufficiencyStatus
from reasoning.reflection_advisor import ReflectionAction, ReflectionAdvice
from reflection.reflection_context import ReflectionContext
from reflection.reflection_engine import ReflectionEngine
from reflection.reflection_repository import ReflectionRepository
from runtime.contracts import ExecutionProfile
from runtime.prima_runtime import PrimaRuntime
from uncertainty.confidence_signal import OverallConfidence
from uncertainty.execution_gate import ExecutionDecision, UncertaintyGate
from uncertainty.uncertainty_types import DecisionType, UncertaintyBand
from workflow.answer_generation import GenerationOutcome, GenerationResult
from workflow.correction_loop import CorrectionLoop
from workflow.execution_context import ExecutionContext
from workflow.prima_workflow import ExecutionDecisionController, _supported_reflection_query
from world.prediction_result import PredictionResult
from world.world_state import WorldState


class Client:
    provider_name = "test"

    def chat(self, **kwargs):
        raise AssertionError("Replay-only checks cannot generate")


def conversation():
    return LoCoMoAdapter().adapt([{"sample_id": "conv", "conversation": {"speaker_a": "A", "speaker_b": "B",
        "session_1_date_time": "2023-08-20", "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "A fair opened in August 2021."},
        {"speaker": "B", "dia_id": "D1:2", "text": "Nice weather."}]}, "qa": []}])[0]


def test_fact_free_descriptions_cannot_fill_any_factual_pool_or_inflate_stage_ids():
    repository = InMemoryMemoryRepository()
    for index in range(45):
        repository.add(MemoryNote.create("I have 4 related memories about maya, august.", note_id=f"summary{index}",
            memory_type=MemoryType.SEMANTIC, embedding=(1.0, 0.0),
            context={"source": "memory_evolution", "type": "semantic_abstraction", "source_turn_id": "D1:1"}))
    repository.add(MemoryNote.create("Maya visited her aunt in August.", memory_type=MemoryType.SEMANTIC,
        memory_level=MemoryLevel.SEMANTIC_ABSTRACTION, note_id="fact", embedding=(0.8, 0.2),
        context={"source": "document_ingestion", "source_turn_id": "D1:2"}))
    controller = RetrievalController(repository, strategy_names=("dense", "sparse"), reranker=Reranker(backend="disabled"))
    response = controller.retrieve(RetrievalRequest("Maya August", query_embedding=(1.0, 0.0), profile="simple_rag", top_k=1))
    assert [item.note.id for item in response.results] == ["fact"]
    assert all(row["id"] == "fact" for key in ("dense_candidates", "sparse_candidates", "hybrid_candidates") for row in response.diagnostics[key])
    assert len(repository.list()) == 46


def test_other_conversations_cannot_fill_a_factual_pool_before_own_source():
    repository = InMemoryMemoryRepository()
    for index in range(45):
        repository.add(MemoryNote.create("Maya visited her aunt in August.", note_id=f"other{index}",
            memory_type=MemoryType.SEMANTIC, embedding=(1.0, 0.0),
            context={"source_conversation_id": "other", "source_turn_id": f"D1:{index + 1}"}))
    repository.add(MemoryNote.create("Maya visited her aunt in August.", note_id="own",
        memory_type=MemoryType.SEMANTIC, embedding=(0.8, 0.2),
        context={"source_conversation_id": "own", "source_turn_id": "D1:1"}))
    controller = RetrievalController(repository, strategy_names=("dense", "sparse"), reranker=Reranker(backend="disabled"))
    result = controller.retrieve(RetrievalRequest("Maya August", query_embedding=(1.0, 0.0),
        profile="simple_rag", session_id="own", top_k=1))
    assert [item.note.id for item in result.results] == ["own"]


def test_only_verified_generated_descriptions_are_marked_and_lineage_is_retained(monkeypatch):
    repository = InMemoryMemoryRepository()
    for index in range(2):
        repository.add(MemoryNote.create("Maya visited her aunt in August.", note_id=f"parent{index}",
            memory_type=MemoryType.EPISODIC, salience_score=.9))
    monkeypatch.setattr("memory.evolution.memory_evolution_engine.GraphReasoningEngine.detect_communities",
                        lambda self: [["node_parent0", "node_parent1"]])
    engine = MemoryEvolutionEngine(repository, GraphRepository())
    generated = engine.evolve().evolved_memories[0]
    assert not factual_evidence_eligible(generated)
    assert generated.lineage.parent_ids == ("parent0", "parent1")
    class FactualSummary(SemanticAbstractionEngine):
        def summarize_cluster(self, notes, max_words=40):
            return "Maya visited her aunt in August."
    factual = MemoryEvolutionEngine(repository, GraphRepository(), abstraction_engine=FactualSummary()).evolve().evolved_memories[0]
    assert factual_evidence_eligible(factual) and factual.lineage.parent_ids == generated.lineage.parent_ids


def test_source_factual_candidate_retains_neutral_dates_but_not_timestamp_headers():
    repository = InMemoryMemoryRepository()
    baseline = MemoryImportanceEngine(repository)
    candidate = MemoryImportanceEngine(repository, policy="source_factual_candidate")
    provenance = {"source_timestamp": "2023-08-20", "source_turn_id": "D1:1"}
    text = "[2023-08-20] A: The fair opened in August 2021."
    assert not baseline.decide(text, source_provenance=provenance).stored
    retained = candidate.decide(text, source_provenance=provenance)
    assert retained.stored and retained.reason == "source_grounded_factual_retention_candidate"
    assert not candidate.decide(text).stored
    assert not candidate.decide("[2023-08-20] A: Nice weather.", source_provenance=provenance).stored


def test_replay_admission_is_logged_once_including_rejections_and_without_question_duplication(tmp_path):
    runtime = PrimaRuntime(llm_client=Client(), memory_repository=InMemoryMemoryRepository(),
                           maintenance_enabled=False, log_path=tmp_path / "runtime.log")
    runtime.memory_importance_engine.config = MemoryImportanceConfig(threshold=1.0)
    logs, questions = [], []
    asyncio.run(_run_conversation(conversation(), runtime, ExecutionProfile.PRIMA_FULL, IngestionPolicy.NORMAL_PRIMA,
        GenerationConfig(model="fake", provider="test"), 5, 20, MaintenanceMode.DISABLED, set(), questions.append,
        admission_checkpoint=logs.append))
    assert questions == [] and len(logs) == 1
    assert len(logs[0]["turns"]) == 2 and logs[0]["complete"]
    for row in logs[0]["turns"]:
        assert row["created_memory_ids"] == [] and row["admission"]["stored"] is False
        assert {"novelty", "emotion", "relevance", "recurrence", "reflection", "threshold", "reason"} <= row["admission"].keys()


@pytest.mark.parametrize("policy", [IngestionPolicy.NORMAL_PRIMA, IngestionPolicy.CONTROLLED_DOCUMENT])
def test_historical_source_replay_is_idempotent_and_conversation_scoped(tmp_path, policy):
    runtime = PrimaRuntime(llm_client=Client(), runtime_config=RuntimeConfig(admission_policy="source_factual_candidate"),
        memory_repository=InMemoryMemoryRepository(), maintenance_enabled=False, log_path=tmp_path / "runtime.log")
    turn = conversation().turns[0]
    generation = GenerationConfig(model="fake", provider="test")
    for _ in range(2):
        assert runtime.execute_sync(_turn_request("conv", turn, policy, generation)).status.value == "completed"
    assert len(runtime.memory_repository.list()) == 1
    runtime.execute_sync(_turn_request("another", turn, policy, generation))
    assert len(runtime.memory_repository.list()) == 2


def test_gate_overrules_heuristic_recommendation_and_preserves_terminal_precedence():
    estimate = SimpleNamespace(uncertainty=.65, confidence=.4, signals=())
    gate = UncertaintyGate()
    world = SimpleNamespace(risk_score=.1)
    assert gate.evaluate(estimate, world, reflection_enabled=False).suppression_reason == "reflection_off"
    assert gate.evaluate(estimate, world, reflection_budget_available=False).decision is ExecutionDecision.CONTINUE
    assert gate.evaluate(estimate, SimpleNamespace(risk_score=.9), reflection_enabled=False).decision is ExecutionDecision.ABSTAIN
    assert gate.evaluate(estimate, world, clarification_required=True).decision is ExecutionDecision.ASK_FOR_CLARIFICATION
    assert gate.reflection_suppression(enabled=True, remaining_seconds=1, remaining_calls=0, interventions=0, max_interventions=1) == "max_llm_calls"


def test_no_progress_reflection_needs_a_supported_untried_query_and_real_budget():
    question = "Which hospital treated Maya's friend last year?"
    evidence = EvidenceItem("e", "Maya's friend is Cedar.", "source", "episodic", .5, 0, question)
    context = ExecutionContext(question, reasoning_result=AnswerResult("", SufficiencyStatus.NEED_MORE_EVIDENCE,
        .2, (evidence,), 1, "duplicate_query", ReasoningBudget()),
        retrieval_hops=[{"query": question}], request_budget=RequestBudget(max_llm_calls=1),
        metadata={"task_kind": "factual_qa", "reflection_enabled": True, "max_retrieval_calls": 3})
    context.uncertainty = OverallConfidence(.35, .65, (.35, .35), UncertaintyBand.HIGH, (), DecisionType.REFLECT, ())
    state = WorldState()
    context.world_prediction = PredictionResult(state, state, (), (), (), .1, .1)
    supported = _supported_reflection_query(context)
    assert supported == f"{question} Source bridge: Cedar"
    decision = asyncio.run(ExecutionDecisionController().execute(context))
    assert decision.decision is ExecutionDecision.REFLECT
    context.retrieval_hops.append({"query": supported})
    decision = asyncio.run(ExecutionDecisionController().execute(context))
    assert decision.decision is ExecutionDecision.CONTINUE
    assert context.metadata["reflection_suppressed_reason"] == "no_remaining_supported_work"
    context.retrieval_hops.pop()
    context.request_budget.retrieval_calls = 3
    decision = asyncio.run(ExecutionDecisionController().execute(context))
    assert decision.decision is ExecutionDecision.CONTINUE
    assert context.metadata["reflection_suppressed_reason"] == "no_remaining_supported_work"


def test_benchmark_default_is_explicit_reflection_off_and_resets_preserve_policy(tmp_path):
    runtime = PrimaRuntime(mode="benchmark", llm_client=Client(), memory_backend="in_memory", maintenance_enabled=False,
                           runtime_config=RuntimeConfig(admission_policy="source_factual_candidate"), log_path=tmp_path / "runtime.log")
    assert runtime.runtime_manifest()["reflection_enabled"] is False
    assert runtime.reasoning_controller.reflection_advisor is None
    runtime.reset()
    assert runtime.memory_importance_engine.policy == "source_factual_candidate" and not runtime.reflection_enabled


def test_read_only_reflection_never_writes_to_either_memory_store():
    memory = InMemoryMemoryRepository()
    repository = ReflectionRepository(memory)
    result = ReflectionEngine(reflection_repository=repository, trigger_threshold=0).evaluate(
        ReflectionContext("Check source", failure_metadata={"reasoning_event": True, "reason": "low support"}, read_only=True))
    assert result.should_reflect and repository.reflection_history() == [] and memory.list() == []


def test_rejected_corrected_citations_restore_original_output_and_prompt():
    evidence = EvidenceItem("e", "Maya visited her aunt.", "old", "episodic", .9, 0, "")
    reasoning = AnswerResult("", SufficiencyStatus.SUFFICIENT, .9, (evidence,), 1, "sufficient", ReasoningBudget())
    original = GenerationResult(GenerationOutcome.ANSWERED, text="Her aunt", selected_source_ids=("old",), raw_output="original")
    context = ExecutionContext("Who?", generation_result=original, reasoning_result=reasoning,
        prompt_evidence=[{"memory_id": "old"}], metadata={"task_kind": "factual_qa", "generation_count": 1})
    loop = CorrectionLoop()
    attempt = loop.review(context, ReflectionAdvice(True, ReflectionAction.REGENERATE, confidence=.9), "post_execution")
    assert attempt.proposal_accepted and not attempt.accepted
    context.generation_result = GenerationResult(GenerationOutcome.ANSWERED, text="Invented", selected_source_ids=("missing",), raw_output="rejected")
    context.prompt_evidence = [{"memory_id": "new"}]
    context.metadata["generation_count"] = 2
    context.output_validation = SimpleNamespace(valid=True)
    loop.complete(context)
    assert attempt.executed and attempt.validated is False and not attempt.accepted
    assert attempt.rejected_output["raw_output"] == "rejected"
    assert context.generation_result is original and context.prompt_evidence == [{"memory_id": "old"}]


def test_paired_resampling_uses_whole_conversations_and_never_promotes_missing_observations():
    values = [(f"one:{index}", "one", 0., 1.) for index in range(50)] + [("two:0", "two", 1., 0.)]
    result = paired_conversation_interval(values, samples=200)
    assert result["conversation_units"] == 2 and result["paired_questions"] == 51
    assert result["bootstrap_95_ci"] == [-1., 1.]
    with pytest.raises(ValueError):
        paired_conversation_interval([("same", "one", 0., float("nan"))])
    base = [{"case_id": "one:q", "conversation_id": "one", "question": "Where?", "expected_answer": "Paris", "prediction": "Rome", "category": "2"}]
    comparison = paired_conversation_comparison(base, [{**base[0], "prediction": "Paris"}], samples=20)
    assert comparison["metrics"]["answerable_official_f1"]["delta"] == 1
    assert comparison["quality_gates_pass"] is False and comparison["automatic_promotion"] is False
    assert paired_conversation_comparison(base, [{**base[0], "prediction": None}], samples=20)["metrics"]["answerable_official_f1"] is None
    with pytest.raises(ValueError, match="source/question identities"):
        paired_conversation_comparison(base, [{**base[0], "reference_metadata": {"answer": "Elsewhere"}}], samples=20)


def test_person2_contract_examples_keep_old_unavailable_and_new_policy_separate():
    root = Path(__file__).parents[1] / "fixtures" / "locomo_review"
    fixture = json.loads((root / "goal4_producer_contract.json").read_text(encoding="utf-8"))
    candidates = json.loads((root / "goal4_final_candidates.json").read_text(encoding="utf-8"))
    assert fixture["admission"]["complete"] and fixture["admission"]["turn_count"] == 2
    assert fixture["admission"]["accepted_cpu_example"]["created_memory_count"] == 1
    assert {"novelty", "emotion", "relevance", "recurrence", "reflection"} <= set(
        fixture["admission"]["accepted_cpu_example"]["component_names"])
    assert fixture["qa"]["memory_count_after"] == fixture["qa"]["memory_count_before"]
    assert fixture["qa"]["reflection_audit"]["engine_executed"] == 0
    assert all(value is None for value in fixture["old_record_missing_fields"].values())
    for key in ("safe_baseline", "unpromoted_native_candidate"):
        config = RuntimeConfig.model_validate(candidates[key]["runtime_config"])
        assert config.reflection_enabled is False and config.admission_policy == "importance_threshold"
    assert candidates["production_promotion"] is False
