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


@pytest.mark.parametrize("persistent", [False, True])
def test_manager_owns_nested_snapshots(tmp_path, persistent):
    manager = JsonStateManager(tmp_path / "state.json") if persistent else InMemoryStateManager()
    nested = {"steps": [["original"]]}
    state = CognitiveState(goal_state=nested, metadata={"extra": nested})
    saved = manager.save("session", state, 0)
    nested["steps"][0].append("input")
    saved.goal["steps"][0].append("returned")
    snapshot = manager.load("session")
    snapshot.metadata["extra"]["steps"][0].append("loaded")
    assert manager.load("session").goal["steps"] == [["original"]]
    assert manager.load("session").metadata["extra"]["steps"] == [["original"]]


@pytest.mark.parametrize("reset", [None, "one", False])
def test_failed_publication_preserves_cache_and_disk(tmp_path, monkeypatch, reset):
    path = tmp_path / "sessions.json"
    manager = JsonStateManager(path)
    original = manager.apply_delta("one", {"task": {"step": 1}}, 0)
    with monkeypatch.context() as fault:

        def fail(*args):
            raise OSError("replace failed")

        fault.setattr(type(path), "replace", fail)
        with pytest.raises(OSError):
            if reset is False:
                manager.apply_delta("one", {"task": {"step": 2}}, 1)
            else:
                manager.reset(reset)
    assert manager.load("one").to_dict() == original.to_dict()
    assert JsonStateManager(path).load("one").to_dict() == original.to_dict()
    assert manager.save("one", original, 1).version == 2


@pytest.mark.parametrize("global_reset", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
def test_reset_invalidates_even_uncommitted_snapshots(tmp_path, global_reset, persistent):
    path = tmp_path / "sessions.json"
    manager = JsonStateManager(path) if persistent else InMemoryStateManager()
    empty = manager.load("one")
    other = manager.load("two")
    first = manager.save("one", empty, 0)
    manager.reset(None if global_reset else "one")
    if persistent:
        manager = JsonStateManager(path)
    with pytest.raises(StateVersionConflict):
        manager.save("one", empty, 0)
    manager.save("one", manager.load("one"), 0)
    with pytest.raises(StateVersionConflict):
        manager.save("one", first, 1)
    if not global_reset:
        manager.save("two", other, 0)


@pytest.mark.parametrize("already_exists", [False, True])
def test_json_encode_failure_can_retry_same_version(tmp_path, already_exists):
    path = tmp_path / "state.json"
    owner = JsonStateManager(path)
    if already_exists:
        owner.apply_delta("s", {"task": {"real": [1]}}, 0)
    before = owner.load("s")
    candidate = owner.load("s")
    candidate.task["unserializable"] = object()
    with pytest.raises(TypeError):
        owner.save("s", candidate, before.version)
    assert owner.load("s").version == JsonStateManager(path).load("s").version == before.version
    assert "unserializable" not in owner.load("s").task
    assert owner.save("s", before, before.version).version == before.version + 1


def test_racing_writers_have_one_owner_and_reset_keeps_disk_consistent(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    path = tmp_path / "state.json"
    owner = JsonStateManager(path)
    state = owner.load("s")
    gate = Barrier(2)

    def write():
        gate.wait(timeout=5)
        try:
            return owner.save("s", state, 0).version
        except StateVersionConflict:
            return "conflict"

    with ThreadPoolExecutor(2) as pool:
        results = [pool.submit(write) for _ in range(2)]
        assert {f.result(timeout=5) for f in results} == {1, "conflict"}

        def reset():
            gate.wait(timeout=5)
            owner.reset("s")

        def save_other():
            gate.wait(timeout=5)
            owner.apply_delta("other", {"goal": {"active": True}}, 0)

        results = [pool.submit(reset), pool.submit(save_other)]
        for result in results:
            result.result(timeout=5)
    reopened = JsonStateManager(path)
    assert reopened.load("s").version == owner.load("s").version == 0
    assert reopened.load("other").to_dict() == owner.load("other").to_dict()


def test_execute_preserves_committed_state_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    manager = JsonStateManager(path)
    original = manager.apply_delta("s", {"goal": {"active": "old"}}, 0)
    runtime = PrimaRuntime(llm_client=ModelClient(), memory_repository=InMemoryMemoryRepository(),
                           state_manager=manager, log_path=tmp_path / "runtime.log")
    def fail(*args):
        raise OSError("state volume unavailable")
    monkeypatch.setattr(type(path), "replace", fail)
    response = asyncio.run(runtime.execute(PrimaRequest(task_kind=TaskKind.CONVERSATION, profile=ExecutionProfile.MODEL_ONLY,
                                                         input_text="Hello", session_id="s")))
    assert response.outcome.value == "failed"
    assert manager.load("s").to_dict() == JsonStateManager(path).load("s").to_dict() == original.to_dict()


def test_version_only_delta_cannot_cross_reset_recreation(tmp_path):
    owner = JsonStateManager(tmp_path / "s.json")
    previous = owner.apply_delta("s", {"goal": {"active": "before"}}, 0)
    owner.reset("s")
    fresh = owner.load("s")
    owner.save("s", fresh, 0)
    with pytest.raises(StateVersionConflict):
        owner.apply_delta("s", {"goal": {"active": "stale"}}, previous.version)
    result = owner.apply_delta("s", {"goal": {"active": "new"}}, 1, expected_generation=fresh.generation)
    assert result.goal["active"] == "new"
