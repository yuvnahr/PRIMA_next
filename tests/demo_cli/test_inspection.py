from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from io import StringIO
from os import terminal_size

import pytest

from demo_cli.commands import dispatch_command
from demo_cli.config import TraceMode
from demo_cli.diagnostics import adapt_response, component_status, state_rows
from demo_cli.renderer import Renderer
from memory.memory_note import MemoryNote
from memory.memory_types import MemoryType
from runtime import ExecutionProfile, PrimaRuntime, RuntimeComponent, RuntimeDiagnostics
from tests.demo_cli.test_session_cli import _session


def test_inspection_commands_before_turn_are_read_only(monkeypatch) -> None:
    session, provider = _session()
    output = StringIO()
    renderer = Renderer(output=output, no_color=True)

    async def unexpected_execute(*args, **kwargs):
        pytest.fail("Inspection must not execute a request")

    monkeypatch.setattr(PrimaRuntime, "execute", unexpected_execute)
    monkeypatch.setattr(session.runtime.retrieval_controller, "retrieve", lambda *args: pytest.fail("No retrieval"))
    monkeypatch.setattr(session.runtime.affect_engine, "process", lambda *args, **kwargs: pytest.fail("No affect analysis"))

    async def run():
        for command in ("/state", "/memory", "/plan", "/world", "/profile", "/profile model", "/clear", "/help", ""):
            assert await dispatch_command(command, session, renderer)

    asyncio.run(run())
    text = output.getvalue()
    for heading in ("COGNITIVE STATE", "MEMORY", "CURRENT PLAN", "WORLD MODEL", "PROFILE"):
        assert heading in text
    assert text.count("No turn response is available yet.") == 2
    assert dict(session.current_state())["Version"] == "0"
    assert dict(session.stored_memories())["Stored memories"] == "0"
    assert not provider.requests and session.latest_response is None


@pytest.mark.parametrize("command", ["/state", "/memory", "/plan", "/world", "/clear", "/reset", "/profile"])
def test_invalid_extra_arguments_do_not_change_session(command) -> None:
    session, provider = _session()
    runtime, session_id, profile = session.runtime, session.session_id, session.current_profile
    output = StringIO()
    extra = " prima extra" if command == "/profile" else " extra"
    assert asyncio.run(dispatch_command(command + extra, session, Renderer(output=output)))
    assert "Usage:" in output.getvalue()
    assert session.runtime is runtime and session.session_id == session_id and session.current_profile is profile
    assert not provider.requests


def test_unknown_profile_preserves_current_profile_and_lists_choices() -> None:
    session, provider = _session()
    output = StringIO()
    assert asyncio.run(dispatch_command("/profile invalid", session, Renderer(output=output)))
    assert session.current_profile is ExecutionProfile.PRIMA_FULL
    assert "Unknown profile 'invalid'" in output.getvalue() and "prima, rag, model" in output.getvalue()
    assert not provider.requests


def test_current_state_uses_manager_not_last_response_and_bounds_fields() -> None:
    session, _ = _session()

    async def run():
        await session.start()
        try:
            response = await session.send("Please remember that I prefer concise explanations.")
            state = session.runtime.state_manager.load(session.session_id)
            assert state.version == response.diagnostics.state_version == 1
            rows = dict(session.current_state())
            assert rows["Emotion"] == state.emotional_state.dominant_emotion
            assert rows["Arousal"] == f"{state.emotional_state.current_pad.arousal:.3f}"
            assert rows["Affect confidence"] == f"{state.emotional_state.confidence:.3f}"
            assert int(dict(session.stored_memories())["Stored memories"]) > 0
            assert "concise explanations" in str(session.stored_memories())
            session.runtime.state_manager.apply_delta(session.session_id, {
                "goal": {"objective": "current-goal " + "x" * 1000},
                "confidence": {"overall_confidence": 0.42},
                "task": {"status": "ready", "metadata": {"private": "do not print"}},
            }, expected_version=state.version)
            current = dict(session.current_state())
            assert current["Version"] == "2" and response.diagnostics.state_version == 1
            assert current["Goal"].startswith("objective=current-goal") and len(current["Goal"]) <= 240
            assert current["Confidence"] == "overall_confidence=0.420" and "private" not in str(current)
            state.emotional_state = None
            missing = dict(state_rows(state, session.session_id))
            assert missing["Emotion"] == missing["Arousal"] == "n/a"
        finally:
            await session.close()

    asyncio.run(run())


def test_stored_memory_is_bounded_recent_and_session_isolated(monkeypatch) -> None:
    session, provider = _session()
    other, _ = _session()
    repository = session.runtime.memory_repository
    start = datetime(2026, 11, 15, tzinfo=timezone.utc)
    for index in range(9):
        repository.add(MemoryNote.create(
            f"memory {index} " + "x" * 1000, MemoryType.EPISODIC, embedding=(987654321.0,),
            note_id=f"note-{index}", timestamp=start + timedelta(minutes=index), salience_score=0.5,
            context={"source_session_id": session.session_id, "private": "hidden-metadata"},
        ))
    repository.add(MemoryNote.create(
        "foreign secret", MemoryType.EPISODIC, embedding=(1.0,), note_id="foreign",
        context={"source_session_id": "other-session"}, timestamp=start + timedelta(days=1),
    ))
    repository.add(MemoryNote.create(
        "derived semantic note", MemoryType.SEMANTIC, embedding=(1.0,), note_id="derived",
        context={"source": "memory_evolution"}, timestamp=start + timedelta(hours=1),
    ))
    monkeypatch.setattr(repository, "query", lambda *args, **kwargs: pytest.fail("Inspection must list, not retrieve"))
    rows = session.stored_memories()
    assert dict(rows)["Stored memories"] == "10"
    entries = [(label, value) for label, value in rows if label.startswith("[")]
    assert len(entries) == 5 and entries[0][1] == "derived semantic note"
    assert entries[1][1].startswith("memory 8")
    assert all(len(value) <= 240 for _, value in entries)
    assert ("More", "... 5 more") in rows
    assert "foreign" not in str(rows) and "hidden-metadata" not in str(rows) and "987654321" not in str(rows)
    assert dict(other.stored_memories())["Stored memories"] == "0" and not provider.requests


def test_profile_switches_keep_runtime_provider_session_and_use_latest_outputs(monkeypatch) -> None:
    session, provider = _session()
    runtime, client, config, session_id = session.runtime, session.runtime.llm_client, session.config, session.session_id
    requests = []
    execute = PrimaRuntime.execute

    async def record(self, request):
        requests.append(request)
        return await execute(self, request)

    monkeypatch.setattr(PrimaRuntime, "execute", record)

    async def run():
        await session.start()
        try:
            for alias, profile in (("prima", ExecutionProfile.PRIMA_FULL), ("model", ExecutionProfile.MODEL_ONLY),
                                   ("rag", ExecutionProfile.SIMPLE_RAG), ("prima", ExecutionProfile.PRIMA_FULL)):
                calls = len(provider.requests)
                assert await dispatch_command(f"/profile {alias}", session, Renderer(output=StringIO()))
                assert len(provider.requests) == calls
                response = await session.send("Please remember that I prefer concise explanations.")
                assert requests[-1].profile is response.profile is profile
                assert session.runtime is runtime and runtime.llm_client is client and client.provider is provider
                assert session.config is config and session.session_id == session_id
                assert dict(session.status_rows())["Profile"] == profile.value
                assert dict(session.latest_diagnostics.summary)["Profile"] == profile.value
                if profile is ExecutionProfile.PRIMA_FULL:
                    assert response.output_data["plan"] and response.output_data["world_prediction"]
                    assert response.output_data["uncertainty"] and response.output_data["memory_admission"]
                    assert component_status(response, RuntimeComponent.PLANNER) == "executed"
                    assert dict(session.latest_plan())["Goal"] == response.output_data["plan"]["goal"][:240]
                    world = response.output_data["world_prediction"]
                    assert dict(session.latest_world())["Risk score"] == f"{world['risk_score']:.3f}"
                    assert dict(session.latest_world())["Failure probability"] == f"{world['failure_probability']:.3f}"
                else:
                    assert component_status(response, RuntimeComponent.PLANNER) == "skipped by profile"
                    assert "Planning was not executed" in str(session.latest_plan())
                    assert "World simulation was not executed" in str(session.latest_world())
                    assert "risk" not in str(session.latest_world()).lower()
            assert [request.profile for request in requests] == [
                ExecutionProfile.PRIMA_FULL, ExecutionProfile.MODEL_ONLY,
                ExecutionProfile.SIMPLE_RAG, ExecutionProfile.PRIMA_FULL,
            ]
            assert requests[0].session_id == requests[-1].session_id
            assert dict(session.latest_diagnostics.summary)["State"] == "v3 -> v4"
        finally:
            await session.close()

    asyncio.run(run())


def test_plan_world_bounds_use_only_public_response_and_commands_do_not_execute(monkeypatch) -> None:
    session, provider = _session()

    async def run():
        await session.start()
        try:
            response = await session.send("Please remember that I prefer concise explanations.")
            data = json.loads(response.to_json())["output_data"]
            data["plan"].update(actions=[{"description": "a" * 1000}] * 23, action_count=23,
                                constraints=[{"description": "c" * 1000}] * 22, constraint_count=22)
            world = data["world_prediction"]
            world["state_transitions"] = world["state_transitions"][:1] * 15
            world["warnings"] = ["w" * 1000] * 16
            session.latest_response = response.model_copy(update={"output_data": data})
            plan, prediction = session.latest_plan(), session.latest_world()
            assert len([row for row in plan if row[0].startswith("Action ")]) == 10
            assert len([row for row in plan if row[0].startswith("Constraint ")]) == 10
            assert ("Action", "... 13 more") in plan and ("Constraint", "... 12 more") in plan
            assert len([row for row in prediction if row[0].startswith("Predicted effect ")]) == 10
            assert len([row for row in prediction if row[0].startswith("Warning ")]) == 10
            assert ("Warnings", "... 6 more") in prediction
            assert all(len(value) <= 240 for _, value in (*plan, *prediction))
            calls = len(provider.requests)

            async def unexpected(*args, **kwargs):
                pytest.fail("No execution during inspection")

            monkeypatch.setattr(PrimaRuntime, "execute", unexpected)
            for command in ("/state", "/memory", "/plan", "/world"):
                assert await dispatch_command(command, session, Renderer(output=StringIO()))
            assert len(provider.requests) == calls
        finally:
            await session.close()

    asyncio.run(run())


def test_clear_preserves_all_conversation_data(monkeypatch) -> None:
    session, provider = _session()
    renderer = Renderer(output=StringIO(), no_color=True)
    cleared = []
    monkeypatch.setattr(renderer, "clear", lambda: cleared.append(True))

    async def run():
        await session.start()
        try:
            await session.send("Please remember that I prefer concise explanations.")
            runtime, response, diagnostics = session.runtime, session.latest_response, session.latest_diagnostics
            before = session.session_id, session.current_profile, session.current_state(), session.stored_memories(), len(provider.requests)
            assert await dispatch_command("/clear", session, renderer)
            assert cleared == [True] and session.runtime is runtime
            assert session.latest_response is response and session.latest_diagnostics is diagnostics
            after = session.session_id, session.current_profile, session.current_state(), session.stored_memories(), len(provider.requests)
            assert before == after
        finally:
            await session.close()

    asyncio.run(run())


def test_exception_without_response_does_not_expose_previous_plan_or_world(monkeypatch) -> None:
    session, provider = _session()

    async def run():
        await session.start()
        try:
            await session.send("Please remember that I prefer concise explanations.")
            assert session.latest_response.output_data["plan"]
            execute = session.runtime.execute
            calls = len(provider.requests)

            async def broken(request):
                raise RuntimeError("No response returned")

            monkeypatch.setattr(session.runtime, "execute", broken)
            with pytest.raises(RuntimeError, match="No response returned"):
                await session.send("Next turn")
            assert session.latest_response is session.latest_diagnostics is None
            assert "No turn response is available" in str(session.latest_plan())
            assert "No turn response is available" in str(session.latest_world())
            assert len(provider.requests) == calls and dict(session.current_state())["Version"] == "1"
            monkeypatch.setattr(session.runtime, "execute", execute)
            await session.send("Please remember that I prefer slow explanations.")
            assert session.latest_response is not None and session.latest_response.diagnostics.state_version == 2
        finally:
            await session.close()

    asyncio.run(run())


def test_reset_replaces_only_owned_runtime_and_preserves_provider_configuration() -> None:
    session, provider = _session()
    other, _ = _session()
    output = StringIO()

    async def run():
        await session.start()
        try:
            await session.send("Please remember that I prefer concise explanations.")
            await session.send("What kind of explanations do I prefer?")
            old = session.runtime
            old_id = session.session_id
            client, generation = old.llm_client, session.config.generation
            assert old.state_manager.load(old_id).version == 2 and old.memory_repository.list()
            foreign = old.state_manager.load("foreign-state")
            foreign.goal_state["objective"] = "outside the active session"
            old.state_manager.save("foreign-state", foreign, 0)
            session.set_profile("rag")
            session.trace_mode = TraceMode.FULL
            calls = len(provider.requests)
            assert await dispatch_command("/reset", session, Renderer(output=output))
            assert session.runtime is not old and not old.maintenance_supervisor.running
            assert old.maintenance_supervisor.diagnostics()["queue_size"] == 0
            assert session.runtime.maintenance_supervisor.running
            assert session.runtime.llm_client is client and client.provider is provider
            assert session.config.generation is generation and session.current_profile is ExecutionProfile.SIMPLE_RAG
            assert session.trace_mode is TraceMode.FULL and session.session_id != old_id
            assert session.latest_response is session.latest_diagnostics is None
            assert dict(session.current_state())["Version"] == "0"
            assert session.runtime.memory_repository.list() == [] and not session.runtime.affect_engine.history.get_recent(50)
            assert len(provider.requests) == calls
            assert old.state_manager.load(old_id).version == 2 and old.memory_repository.list()
            assert old.state_manager.load("foreign-state").goal_state["objective"] == "outside the active session"
            assert dict(other.current_state())["Version"] == "0"
            third = await session.send("Please remember that I prefer slow explanations.")
            assert third.diagnostics.state_version == 1 and third.output_data["retrieval"]["selected_count"] == 0
        finally:
            await session.close()

    asyncio.run(run())
    assert "SESSION RESET" in output.getvalue() and "Previous session" in output.getvalue()
    assert not session.runtime.maintenance_supervisor.running


def test_component_status_distinguishes_failures_skips_missing_and_recovery() -> None:
    from tests.demo_cli.test_diagnostics_renderer import _response

    missing = _response()
    assert component_status(missing, RuntimeComponent.PLANNER) == "unavailable"
    assert component_status(missing, RuntimeComponent.STATE_MANAGER) == "unavailable"
    skipped = _response(diagnostics=RuntimeDiagnostics(skipped_components=(RuntimeComponent.PLANNER,)))
    assert component_status(skipped, RuntimeComponent.PLANNER) == "skipped by profile"
    absent = _response(diagnostics=RuntimeDiagnostics(not_executed_components=(RuntimeComponent.PLANNER,)))
    assert component_status(absent, RuntimeComponent.PLANNER) == "not executed"
    failed = _response(diagnostics=RuntimeDiagnostics(
        trace_events=({"event_type": "phase_failed", "phase": "planning"},),
        not_executed_components=(RuntimeComponent.PLANNER,),
    ))
    assert component_status(failed, RuntimeComponent.PLANNER) == "failed"
    assert dict(adapt_response(failed).summary)["Planning"] == "failed"
    recovered = _response(diagnostics=RuntimeDiagnostics(
        trace_events=({"event_type": "phase_failed", "phase": "planning"},
                      {"event_type": "phase_completed", "phase": "planning"}),
        executed_components=(RuntimeComponent.PLANNER,),
    ))
    assert component_status(recovered, RuntimeComponent.PLANNER) == "executed"
    failed_plan = _response(output_data={"plan": {"status": "failed"}})
    assert component_status(failed_plan, RuntimeComponent.PLANNER) == "failed"
    stale = _response(output_data={"plan": {"status": "failed"}},
                      diagnostics=RuntimeDiagnostics(skipped_components=(RuntimeComponent.PLANNER,)))
    assert component_status(stale, RuntimeComponent.PLANNER) == "skipped by profile"


def test_real_evidence_acquisition_error_is_failed_not_empty_retrieval(monkeypatch) -> None:
    session, _ = _session()

    def unavailable(*args, **kwargs):
        raise ValueError("Retrieval unavailable")

    monkeypatch.setattr(session.runtime.retrieval_controller, "retrieve", unavailable)

    async def run():
        await session.start()
        try:
            response = await session.send("Please remember that I prefer concise explanations.")
            assert response.output_data["sufficiency_status"] == "error"
            assert component_status(response, RuntimeComponent.DENSE_RETRIEVAL) == "failed"
            assert dict(session.latest_diagnostics.summary)["Retrieval"] == "failed"
        finally:
            await session.close()

    asyncio.run(run())


def test_renderer_clear_uses_no_shell_and_respects_terminal_support(monkeypatch) -> None:
    monkeypatch.setattr("demo_cli.renderer.shutil.get_terminal_size", lambda *args: terminal_size((80, 90)))
    output = StringIO()
    renderer = Renderer(output=output, no_color=True)
    renderer.clear()
    assert output.getvalue().isspace()
    assert output.getvalue().count("\n") >= 90
    monkeypatch.setattr(renderer, "ansi", True)
    renderer.clear()
    assert len(output.getvalue()) < 100
