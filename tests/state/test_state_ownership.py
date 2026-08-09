"""Phase 04 state and repository ownership checks."""

from __future__ import annotations

import asyncio
import json

import pytest

from config.runtime_mode import RuntimeMode
from llm.llm_types import LLMResponse
from memory.maintenance.importance_types import MemoryImportanceConfig
from memory.maintenance.memory_importance import MemoryImportanceEngine
from memory.memory_repository import ChromaMemoryRepository, InMemoryMemoryRepository
from memory.memory_types import MemoryType
from runtime import ExecutionProfile, PrimaRequest, PrimaRuntime, TaskKind
from state import CognitiveState, InMemoryStateManager, JsonStateManager, StateVersionConflict
from workflow.execution_context import ExecutionContext
from workflow.prima_workflow import MemoryCommitController


class ModelClient:
    provider_name = "test"

    def chat(self, **kwargs: object) -> LLMResponse:
        if kwargs.get("response_format"):
            return LLMResponse(json.dumps({"answer": "known", "evidence": [], "insufficient_information": False}))
        return LLMResponse("generated")


def test_typed_state_roundtrip_is_versioned() -> None:
    state = CognitiveState(goal={"active": "answer"}, task={"kind": "qa"})
    restored = CognitiveState.from_dict(state.to_dict())
    assert restored.goal["active"] == "answer"
    assert restored.task["kind"] == "qa"
    assert restored.version == 0 and restored.created_at.tzinfo is not None


def test_state_manager_isolates_sessions_and_rejects_stale_writes(tmp_path) -> None:
    manager = InMemoryStateManager()
    first = manager.apply_delta("one", {"goal": {"active": "one"}}, 0)
    assert manager.load("two").goal == {}
    with pytest.raises(StateVersionConflict):
        manager.save("one", first, 0)

    persistent = JsonStateManager(tmp_path / "state.json")
    persistent.apply_delta("one", {"task": {"step": 1}}, 0)
    assert JsonStateManager(tmp_path / "state.json").load("one").task["step"] == 1


def test_conversation_and_qa_share_session_state(tmp_path) -> None:
    manager = InMemoryStateManager()
    runtime = PrimaRuntime(
        llm_client=ModelClient(),
        memory_repository=InMemoryMemoryRepository(),
        state_manager=manager,
        log_path=tmp_path / "runtime.log",
    )
    conversation = PrimaRequest(
        task_kind=TaskKind.CONVERSATION,
        profile=ExecutionProfile.PRIMA_FULL,
        input_text="I feel joyful.",
        session_id="shared",
    )
    qa = PrimaRequest(
        task_kind=TaskKind.FACTUAL_QA,
        profile=ExecutionProfile.MODEL_ONLY,
        input_text="What is known?",
        session_id="shared",
    )
    isolated = PrimaRequest(
        task_kind=TaskKind.FACTUAL_QA,
        profile=ExecutionProfile.MODEL_ONLY,
        input_text="What is known?",
        session_id="other",
    )

    first = asyncio.run(runtime.execute(conversation))
    second = asyncio.run(runtime.execute(qa))
    asyncio.run(runtime.execute(isolated))

    assert first.diagnostics.state_version == 1 and second.diagnostics.state_version == 2
    assert manager.load("shared").emotional.dominant_emotion != "neutral"
    assert manager.load("other").version == 1


def test_runtime_modes_prevent_silent_repository_fallback(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Production preflight"):
        PrimaRuntime(mode=RuntimeMode.PRODUCTION, llm_client=ModelClient(), log_path=tmp_path / "p.log")
    with pytest.raises(RuntimeError, match="declare"):
        PrimaRuntime(mode=RuntimeMode.BENCHMARK, llm_client=ModelClient(), log_path=tmp_path / "b.log")

    benchmark = PrimaRuntime(
        mode=RuntimeMode.BENCHMARK, memory_backend="in_memory", llm_client=ModelClient(), log_path=tmp_path / "ok.log"
    )
    manifest = benchmark.runtime_manifest()
    assert manifest["schema_version"] == "1.0"
    assert manifest["runtime_mode"] == "benchmark" and manifest["memory_repository"] == "in_memory"

    other_benchmark = PrimaRuntime(
        mode=RuntimeMode.BENCHMARK,
        memory_backend="in_memory",
        llm_client=ModelClient(),
        log_path=tmp_path / "other.log",
    )
    request = PrimaRequest(
        task_kind=TaskKind.CONVERSATION,
        profile=ExecutionProfile.MODEL_ONLY,
        input_text="Keep benchmark state isolated.",
        session_id="same-id",
    )
    assert asyncio.run(benchmark.execute(request)).diagnostics.state_version == 1
    assert asyncio.run(other_benchmark.execute(request)).diagnostics.state_version == 1
    with pytest.raises(RuntimeError, match="Only test mode"):
        benchmark.reset(preserve_repository=False)
    assert benchmark.state_manager.load("same-id").version == 1

    production = PrimaRuntime(
        mode=RuntimeMode.PRODUCTION,
        memory_repository=ChromaMemoryRepository(str(tmp_path / "memory")),
        state_manager=JsonStateManager(tmp_path / "production-state.json"),
        llm_client=ModelClient(),
        log_path=tmp_path / "production.log",
    )
    assert production.runtime_manifest()["memory_persistent"] is True


def test_conversation_memory_policy_records_user_and_assistant_outcomes() -> None:
    repository = InMemoryMemoryRepository()
    controller = MemoryCommitController(
        repository,
        MemoryImportanceEngine(repository, MemoryImportanceConfig(threshold=0.0)),
    )
    context = ExecutionContext(user_input="Remember my important research goal.")
    context.output = {"outcome": "answered", "text": "I will remember that research goal."}
    context.metadata.update({"session_id": "session", "turn_id": "turn"})

    result = asyncio.run(controller.execute(context))

    episodic = repository.list(MemoryType.EPISODIC)
    assert [note.context["role"] for note in episodic] == ["user", "assistant"]
    assert result["admission"]["record_count"] == 2
