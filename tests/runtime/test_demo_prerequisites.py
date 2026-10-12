from __future__ import annotations

import json
from dataclasses import replace

import pytest

from affect.affect_engine import DynamicAffectEngine
from affect.emotion_profile import EmotionProfile
from config.runtime_config import RuntimeConfig
from config.settings import Settings
from llm.generation_config import GenerationConfig
from llm.llm_client import LLMClient
from llm.llm_types import LLMResponse
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_confidence import RetrievalConfidence
from memory.retrieval.retrieval_controller import RetrievalResponse
from memory.retrieval.retrieval_result import RetrievalResult
from planning import PlanningContext, TaskPlanner
from reasoning.models import AnswerResult, EvidenceItem, ReasoningBudget, SufficiencyStatus
from runtime import ExecutionOptions, ExecutionProfile, PrimaRequest, PrimaResponse, PrimaRuntime, TaskKind
from runtime.route_profiles import select_route
from state.cognitive_state import CognitiveState
from workflow.answer_generation import AnswerGenerationController
from workflow.execution_context import ExecutionContext


class RecordingClient:
    provider_name = "test"

    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse("Generated response.")


class FixedClassifier:
    def __init__(self, confidence=0.8):
        self.confidence = confidence

    def classify(self, text):
        return EmotionProfile.from_scores({"trust": 0.8}, confidence=self.confidence)


def _runtime(tmp_path, client=None, **kwargs):
    return PrimaRuntime(
        llm_client=client or RecordingClient(),
        affect_engine=DynamicAffectEngine(classifier=FixedClassifier()),
        runtime_config=RuntimeConfig(),
        memory_repository=kwargs.pop("memory_repository", InMemoryMemoryRepository()),
        maintenance_enabled=False,
        log_path=tmp_path / "runtime.log",
        **kwargs,
    )


def _trusted_context(prompt):
    return json.loads(prompt.split("<TRUSTED_RUNTIME_CONTEXT>\n", 1)[1].split("\n</TRUSTED_RUNTIME_CONTEXT>", 1)[0])


@pytest.mark.parametrize("confidence", [0.0, 0.15, 0.9, None])
def test_generation_context_contains_compact_affect_and_plain_state(confidence) -> None:
    state = CognitiveState(goal={"objective": "finish"}, task={"kind": "conversation"}, confidence={"score": 0.7})
    context = ExecutionContext("help me", cognitive_state=state)
    if confidence is not None:
        context.affect_update = DynamicAffectEngine(classifier=FixedClassifier(confidence)).process(
            context.user_input, cognitive_state=state,
        )
    prompt, labels = AnswerGenerationController(RecordingClient())._prompt(context, None, False)
    data = _trusted_context(prompt.user_prompt)
    assert data["state"] == {"goal": {"objective": "finish"}, "task": {"kind": "conversation"}, "confidence": {"score": 0.7}}
    assert labels == {}
    if confidence is None:
        assert data["affect"] is None
    else:
        update = context.affect_update
        assert data["affect"] == {
            "dominant_emotion": "trust", "confidence": confidence,
            "valence": update.profile.valence, "arousal": update.profile.arousal,
            "dominance": update.profile.dominance,
            "current_pad": update.emotional_state.current_pad.to_dict(),
            "emotional_stability": update.emotional_state.emotional_stability,
            "emotional_volatility": update.emotional_state.emotional_volatility,
            "salience_score": update.salience_score, "dissonance_score": update.dissonance_score,
        }
        assert "memory_metadata" not in data["affect"]
    assert "confidence is low or affect is missing" in prompt.system_policy
    assert "never a fact about the user" in prompt.system_policy


def test_conversation_persona_policy_and_factual_policy_stay_separate() -> None:
    policy = AnswerGenerationController._system_prompt(False)
    for guidance in ("consistent, grounded", "continuity", "adapt subtly", "uncertain adaptive control", "actual request", "unless explicitly asked"):
        assert guidance in policy
    factual = AnswerGenerationController._system_prompt(True)
    assert "Return only JSON" in factual and "cite selected" in factual


def test_full_runtime_groq_generation_uses_affect_and_exposes_real_plan_and_memories(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_post(url, payload, headers, timeout):
        calls.append(payload)
        return {"choices": [{"message": {"content": "Generated response."}}], "usage": {"total_tokens": 12}}

    monkeypatch.setattr("llm.provider.post_json", fake_post)
    settings = Settings(_env_file=None, groq_api_key="fixture-key")
    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("I prefer concise answers.", MemoryType.EPISODIC, note_id="preference"))
    runtime = _runtime(
        tmp_path, LLMClient(provider_name="groq", settings=settings), memory_repository=repository,
        generation_config=GenerationConfig(model="openai/gpt-oss-120b", provider="groq", max_output_tokens=512),
    )
    response = runtime.execute_sync(PrimaRequest(
        task_kind=TaskKind.CONVERSATION, profile=ExecutionProfile.PRIMA_FULL,
        input_text="Please explain my preference for concise answers.", session_id="demo",
        options=ExecutionOptions(reasoning_mode="single_pass"),
    ))
    assert response.output_text == "Generated response."
    assert calls and _trusted_context(calls[-1]["messages"][-1]["content"])["affect"]["confidence"] == 0.8
    assert response.diagnostics.provider["name"] == "groq"
    assert response.output_data["generation"]["provider"] == "groq"
    plan = response.output_data["plan"]
    assert plan["goal"] and plan["actions"] and plan["constraints"]
    assert isinstance(plan["confidence"], float) and plan["revision"] >= 0
    retrieval = response.output_data["retrieval"]
    assert retrieval["candidate_count"] >= retrieval["selected_count"] == 1, response.output_data.get("reasoning_errors")
    assert retrieval["final_context_count"] == len(response.evidence) == 1
    assert retrieval["retrieved_memories"][0]["id"] == "preference"
    assert retrieval["retrieved_memories"][0]["memory_type"] == "episodic"
    assert retrieval["final_context_memories"][0]["id"] == "preference"
    assert response.output_data["world_prediction"] and response.output_data["uncertainty"]
    assert response.state_delta.changes["committed"] is True
    assert PrimaResponse.from_json(response.to_json()).to_dict() == response.to_dict()
    assert "fixture-key" not in response.to_json()


@pytest.mark.parametrize("profile", [ExecutionProfile.MODEL_ONLY, ExecutionProfile.SIMPLE_RAG, ExecutionProfile.PRIMA_FULL])
def test_empty_retrieval_and_skipped_planning_serialize_cleanly(tmp_path, profile) -> None:
    response = _runtime(tmp_path).execute_sync(PrimaRequest(
        task_kind=TaskKind.CONVERSATION, profile=profile, input_text="Hello",
        options=ExecutionOptions(reasoning_mode="single_pass"),
    ))
    retrieval = response.output_data["retrieval"]
    assert retrieval["candidate_count"] == retrieval["selected_count"] == retrieval["final_context_count"] == 0
    assert retrieval["retrieved_memories"] == retrieval["final_context_memories"] == []
    if profile is not ExecutionProfile.PRIMA_FULL:
        assert response.output_data["plan"] is None
    assert retrieval["retrieval_confidence"] == (None if profile is ExecutionProfile.MODEL_ONLY else 0.0)
    assert PrimaResponse.from_json(response.to_json()).to_dict() == response.to_dict()


def test_retrieval_distinguishes_selected_memories_from_compressed_context(tmp_path) -> None:
    repository = InMemoryMemoryRepository()
    for index in range(2):
        repository.add(MemoryNote.create("I prefer concise answers.", MemoryType.EPISODIC, note_id=f"preference-{index}"))
    response = _runtime(tmp_path, memory_repository=repository).execute_sync(PrimaRequest(
        task_kind=TaskKind.CONVERSATION, profile=ExecutionProfile.SIMPLE_RAG,
        input_text="Explain my preference for concise answers.",
        options=ExecutionOptions(reasoning_mode="single_pass", top_k=2, max_context_tokens=1),
    ))
    audit = response.output_data["retrieval"]
    assert audit["candidate_count"] == audit["selected_count"] == 2
    assert audit["final_context_count"] == len(response.evidence) == 1
    assert len(audit["retrieved_memories"]) == 2 and len(audit["final_context_memories"]) == 1
    assert len(audit["final_context_memories"][0]["preview"].split()) == 1


@pytest.mark.parametrize("evaluated", [True, False])
def test_public_plan_is_bounded_and_excludes_private_objects(tmp_path, evaluated) -> None:
    runtime = _runtime(tmp_path)
    plan = TaskPlanner().create_plan(PlanningContext("objective"))
    private = {"private": lambda: None, "body": "hidden payload"}
    plan = replace(
        plan, goal=replace(plan.goal, description="g" * 1000, metadata=private),
        actions=(replace(plan.actions[0], description="a" * 1000, inputs=("hidden input",), metadata=private),) * 35,
        constraints=(replace(plan.constraints[0], description="c" * 1000, metadata=private),) * 35,
        metadata=private, revision=2, evaluation=plan.evaluation if evaluated else None,
    )
    context = ExecutionContext("objective", plan=plan, output={"outcome": "answered", "text": "generated"})
    request = PrimaRequest(task_kind=TaskKind.CONVERSATION, profile=ExecutionProfile.PRIMA_FULL, input_text="objective")
    response = runtime._response_from_execution(request, select_route(request.task_kind, request.profile), context, 1.0)
    summary = response.output_data["plan"]
    assert summary["revision"] == 2
    assert summary["confidence"] == (plan.evaluation.confidence if evaluated else plan.goal.confidence)
    assert summary["action_count"] == summary["constraint_count"] == 35
    assert len(summary["actions"]) == len(summary["constraints"]) == 30
    assert len(summary["goal"]) == len(summary["actions"][0]["description"]) == len(summary["constraints"][0]["description"]) == 240
    assert "hidden" not in response.to_json() and "private" not in response.to_json()


def test_retrieval_reports_latest_selection_and_final_context_across_hops(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    selected = tuple(RetrievalResult(
        MemoryNote.create("long memory " * 100, MemoryType.EPISODIC, note_id=f"memory-{index}"), 0.8,
    ) for index in range(35))
    earlier = RetrievalResult(MemoryNote.create("Earlier hop", MemoryType.SEMANTIC, note_id="earlier"), 0.9)
    evidence = (EvidenceItem("unbacked", "External evidence", "external", "memory", 0.7, 0, "query"),)
    evidence = tuple(EvidenceItem(
        f"ev-{index}", result.note.content, result.note.id, "memory", result.score, 0, "query", result=result,
    ) for index, result in enumerate((earlier, *selected))) + evidence
    reasoning = AnswerResult("", SufficiencyStatus.SUFFICIENT, 0.8, evidence, 2, "sufficient", ReasoningBudget())
    context = ExecutionContext(
        "query", reasoning_result=reasoning,
        retrieval_response=RetrievalResponse(selected, RetrievalConfidence(0.82, 0.1, 1.0, 0.8), {
            "hybrid_candidates": [{"id": str(index)} for index in range(40)],
        }),
        output={"outcome": "answered", "text": "generated"},
    )
    request = PrimaRequest(task_kind=TaskKind.CONVERSATION, profile=ExecutionProfile.PRIMA_FULL, input_text="query")
    response = runtime._response_from_execution(request, select_route(request.task_kind, request.profile), context, 1.0)
    audit = response.output_data["retrieval"]
    assert (audit["candidate_count"], audit["selected_count"], audit["final_context_count"]) == (40, 35, 37)
    assert audit["retrieval_confidence"] == 0.82
    assert len(audit["retrieved_memories"]) == len(audit["final_context_memories"]) == 30
    assert all(len(item["preview"]) <= 240 for item in audit["retrieved_memories"])
    assert audit["final_context_memories"][0]["id"] == "earlier"
    assert audit["final_context_memories"][0]["memory_type"] == "semantic"
    assert json.loads(response.to_json())["output_data"]["retrieval"] == audit
    context.reasoning_result = replace(reasoning, evidence_references=evidence[-1:])
    unbacked = runtime._response_from_execution(request, select_route(request.task_kind, request.profile), context, 1.0)
    assert unbacked.output_data["retrieval"]["final_context_memories"][0]["memory_type"] is None
