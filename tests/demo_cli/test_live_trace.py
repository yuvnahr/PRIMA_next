from __future__ import annotations

import asyncio
import threading
from io import StringIO
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from demo_cli import cli
from demo_cli.commands import dispatch_command
from demo_cli.config import TraceMode, resolve_config
from demo_cli.live_trace import MAX_LIVE_EVENTS, LiveTraceObserver
from demo_cli.renderer import Renderer
from events import Event, EventBus, EventType
from monitoring.tracing import Tracing
from runtime import ExecutionProfile, ExecutionStatus, PrimaRequest, PrimaRuntime, TaskKind
from tests.demo_cli.test_persistence import open_session
from tests.demo_cli.test_session_cli import _session, _settings
from tests.workflow.test_phase06_correction_loop import _RegeneratingClient
from workflow.workflow_events import WorkflowEventType
from workflow.workflow_state import WorkflowPhase


@pytest.fixture(autouse=True)
def no_live_provider_or_export(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setattr("llm.provider.post_json", lambda *a, **kw: pytest.fail("No live provider calls"))
    monkeypatch.setattr("config.settings._SETTINGS", _settings())


class RecordingRenderer(Renderer):
    def __init__(self):
        super().__init__(output=StringIO(), no_color=True)
        self.live_rows = []
        self.threads = []

    def live_event(self, marker, phase, detail=""):
        self.live_rows.append((marker, phase, detail))
        self.threads.append(threading.get_ident())
        super().live_event(marker, phase, detail)


def _bus(session):
    return session.runtime.workflow.engine.event_bus.event_bus


def _observers(bus):
    return [item for item in bus.subscribers.all() if item.name == "demo_live_trace"]


def _event(kind, phase=None, *, execution="active", **payload):
    return Event(kind, source="workflow", topic="workflow", execution_id=execution,
                 payload={**payload, **({"phase": phase} if phase is not None else {})})


def test_live_flag_is_opt_in():
    parser = cli.build_parser()
    assert not resolve_config(parser.parse_args(["chat"]), _settings(), {}).live_trace
    assert resolve_config(parser.parse_args(["chat", "--live-trace"]), _settings(), {}).live_trace


@pytest.mark.parametrize("profile", ["prima", "rag", "model"])
def test_real_runtime_output_matches_emitted_events_and_is_live_before_provider(monkeypatch, profile):
    session, provider = _session(["--profile", profile, "--live-trace"])
    renderer = RecordingRenderer()
    session.observe_workflow(renderer)
    thread = threading.get_ident()
    before_generation = []
    send = provider.send
    def checked_send(request):
        before_generation.append("[>>] Answer Generation" in renderer.output.getvalue())
        return send(request)
    monkeypatch.setattr(provider, "send", checked_send)
    async def run():
        await session.start()
        try:
            response = await session.send("Please remember that I prefer concise explanations.")
            assert response.status is ExecutionStatus.COMPLETED
            actual = session.runtime.workflow.engine.event_bus.events
            completed = [e.phase.value for e in actual if e.event_type is WorkflowEventType.PHASE_COMPLETED]
            assert [phase for marker, phase, _ in renderer.live_rows if marker == "[OK]"] == completed
            assert len({e.execution_id for e in actual}) == 1
            assert renderer.threads and set(renderer.threads) == {thread}
            assert before_generation == [True]
            assert "[--] Reflection" not in renderer.output.getvalue()
            assert "concise explanations" not in renderer.output.getvalue()
            before = renderer.output.getvalue()
            await session.runtime.flush_maintenance()
            assert renderer.output.getvalue() == before
            if profile == "prima":
                assert session.runtime.maintenance_supervisor.event_bus.events
        finally:
            await session.close()
    asyncio.run(run())
    assert not _observers(_bus(session))
    assert "\x1b" not in renderer.output.getvalue()


def test_disabled_live_trace_preserves_normal_answer_and_summary(monkeypatch, capsys):
    session, _ = _session(["--profile", "model"])
    inputs = iter(("Please remember my preference.", "/quit"))
    monkeypatch.setattr(cli, "get_settings", _settings)
    monkeypatch.setattr(cli, "DemoSession", lambda *a, **kw: session)
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    assert cli.main(["chat", "--provider", "groq"]) == 0
    output = capsys.readouterr().out
    assert "PRIMA EXECUTION" not in output
    assert "PRIMA > I will keep" in output and "PRIMA TRACE" in output
    assert "Live trace" in output and "off" in output


def test_live_commands_status_and_trace_are_independent():
    session, provider = _session()
    renderer = RecordingRenderer()
    async def run():
        for command in ("/live", "/live on", "/status", "/trace off", "/live off", "/live invalid", "/live on extra"):
            assert await dispatch_command(command, session, renderer)
        assert not session.live_trace and session.trace_mode is TraceMode.OFF
        assert dict(session.status_rows())["Live trace"] == "off"
        assert not provider.requests
        await session.close()
    asyncio.run(run())
    output = renderer.output.getvalue()
    assert "Live workflow trace: on" in output and "Live workflow trace: off" in output
    assert output.count("Usage: /live [on|off]") == 2


@pytest.mark.parametrize("persist", [False, True])
def test_subscriptions_do_not_accumulate_and_reset_changes_bus_and_session(tmp_path, persist):
    session = open_session(tmp_path, persist=persist)
    session.live_trace = True
    renderer = RecordingRenderer()
    session.observe_workflow(renderer)
    old_bus, old_id = _bus(session), session.session_id
    session.observe_workflow(renderer)
    assert len(_observers(old_bus)) == 1
    async def run():
        await session.start()
        try:
            for _ in range(3):
                offset = len(renderer.live_rows)
                response = await session.send("Please remember that I prefer concise explanations.")
                assert response.status is ExecutionStatus.COMPLETED
                completed = sum(row[0] == "[OK]" for row in renderer.live_rows[offset:])
                execution = old_bus.events[-1].execution_id
                emitted = sum(e.execution_id == execution and e.event_type is EventType.WORKFLOW_PHASE_COMPLETED for e in old_bus.events)
                assert completed == emitted and len(_observers(old_bus)) == 1
            await session.reset()
            assert session.live_trace and session.session_id != old_id
            assert not _observers(old_bus) and len(_observers(_bus(session))) == 1
            before = renderer.output.getvalue()
            await old_bus.publish(_event(EventType.WORKFLOW_STARTED, execution="stale"))
            await old_bus.publish(_event(EventType.WORKFLOW_PHASE_COMPLETED, "reflection", execution="stale"))
            assert renderer.output.getvalue() == before
            await session.send("Please remember my new preference.")
            assert renderer.output.getvalue().count("PRIMA EXECUTION") == 4
        finally:
            await session.close()
            await session.close()
    asyncio.run(run())
    assert not _observers(_bus(session))


def test_event_correlation_order_future_phases_and_sensitive_metadata():
    bus, renderer = EventBus(), RecordingRenderer()
    observer = LiveTraceObserver(bus, renderer)
    async def run():
        observer.begin_turn(enabled=True)
        await bus.publish(_event(EventType.WORKFLOW_STARTED))
        await bus.publish(_event(EventType.WORKFLOW_PHASE_COMPLETED, "planning", prompt="private-prompt", api_key="private-key"))
        await bus.publish(_event(EventType.WORKFLOW_PHASE_COMPLETED, "reflection", execution="old-execution"))
        await bus.publish(_event(EventType.WORKFLOW_STARTED, execution="old-execution"))
        await bus.publish(_event(EventType.WORKFLOW_PHASE_COMPLETED, "future_component", embeddings=[1] * 1000))
        await bus.publish(_event(EventType.WORKFLOW_PHASE_RETRY, "planning", retry_count=1, error="private-error"))
        await bus.publish(_event(EventType.WORKFLOW_PHASE_COMPLETED, "planning"))
        observer.end_turn()
        before = renderer.output.getvalue()
        await bus.publish(_event(EventType.WORKFLOW_PHASE_COMPLETED, "affect"))
        assert renderer.output.getvalue() == before
        observer.close()
        observer.close()
    asyncio.run(run())
    assert renderer.live_rows == [("[OK]", "planning", ""), ("[OK]", "future_component", ""),
                                  ("[!!]", "planning", "retry 1"), ("[OK]", "planning", "")]
    output = renderer.output.getvalue()
    assert "Future Component" in output and "private" not in output and "embeddings" not in output
    assert not bus.subscribers.all()


def test_background_and_worker_events_never_print_over_foreground_or_idle_input():
    bus, renderer = EventBus(), RecordingRenderer()
    observer = LiveTraceObserver(bus, renderer)
    async def emit():
        await bus.publish(_event(EventType.WORKFLOW_PHASE_COMPLETED, "affect"))
    async def run():
        observer.begin_turn(enabled=True)
        await bus.publish(_event(EventType.WORKFLOW_STARTED))
        before = renderer.output.getvalue()
        await asyncio.create_task(emit())
        await asyncio.to_thread(lambda: asyncio.run(emit()))
        await bus.publish(Event(EventType.MEMORY_ENCODED, source="memory", topic="memory", execution_id="active"))
        assert renderer.output.getvalue() == before
        observer.end_turn()
        before = renderer.output.getvalue()
        await emit()
        assert renderer.output.getvalue() == before
        observer.close()
    asyncio.run(run())
    assert not renderer.live_rows


def test_event_limit_suppresses_only_display_and_preserves_all_bus_events():
    bus, renderer = EventBus(), RecordingRenderer()
    observer = LiveTraceObserver(bus, renderer)
    received = []
    bus.subscribe(lambda event: received.append(event))
    async def run():
        observer.begin_turn(enabled=True)
        await bus.publish(_event(EventType.WORKFLOW_STARTED))
        for _ in range(MAX_LIVE_EVENTS + 25):
            result = await bus.publish(_event(EventType.WORKFLOW_PHASE_COMPLETED, "planning"))
            assert result.succeeded
        observer.end_turn()
        observer.close()
    asyncio.run(run())
    assert len(renderer.live_rows) == MAX_LIVE_EVENTS
    assert renderer.output.getvalue().count("additional workflow events suppressed") == 1
    assert len(received) == len(bus.events) == MAX_LIVE_EVENTS + 26


def test_real_phase_retry_is_not_deduplicated(monkeypatch):
    session, _ = _session(["--live-trace"])
    renderer = RecordingRenderer()
    session.observe_workflow(renderer)
    controller = session.runtime.workflow.engine.registry.get(WorkflowPhase.PLANNING)
    execute = type(controller).execute
    attempts = []
    async def flaky(self, context):
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("private-transient-detail")
        return await execute(self, context)
    monkeypatch.setattr(type(controller), "execute", flaky)
    async def run():
        try:
            response = await session.send("Please remember my preference.")
            assert response.status is ExecutionStatus.COMPLETED
        finally:
            await session.close()
    asyncio.run(run())
    assert ("[!!]", "planning", "retry 1") in renderer.live_rows
    assert ("[OK]", "planning", "") in renderer.live_rows
    actual = session.runtime.workflow.engine.event_bus.events
    completed = sum(e.phase is WorkflowPhase.PLANNING and e.event_type is WorkflowEventType.PHASE_COMPLETED for e in actual)
    retried = sum(e.phase is WorkflowPhase.PLANNING and e.event_type is WorkflowEventType.PHASE_RETRY for e in actual)
    assert len(attempts) == completed + retried and retried == 1
    assert sum(row == ("[OK]", "planning", "") for row in renderer.live_rows) == completed
    assert "private" not in renderer.output.getvalue()


def test_actual_correction_loop_renders_repeated_generation(tmp_path):
    client = _RegeneratingClient()
    runtime = PrimaRuntime(llm_client=client, log_path=tmp_path / "runtime.log")
    renderer = RecordingRenderer()
    observer = LiveTraceObserver(runtime.workflow.engine.event_bus.event_bus, renderer)
    async def run():
        observer.begin_turn(enabled=True)
        try:
            response = await runtime.execute(PrimaRequest(task_kind=TaskKind.CONVERSATION,
                profile=ExecutionProfile.MODEL_ONLY, input_text="Give a response.", session_id="correction"))
            assert response.output_text == "Corrected answer"
            assert response.output_data["correction_count"] == 1
        finally:
            observer.close()
            await runtime.stop_maintenance()
    asyncio.run(run())
    emitted = [e.phase.value for e in runtime.workflow.engine.event_bus.events if e.event_type is WorkflowEventType.PHASE_COMPLETED]
    shown = [phase for marker, phase, _ in renderer.live_rows if marker == "[OK]"]
    assert shown == emitted and shown.count("answer_generation") == client.calls == 2


def test_phase_failure_safe_final_error_and_next_turn_recovery(monkeypatch, capsys):
    session, _ = _session(["--live-trace"])
    controller = session.runtime.workflow.engine.registry.get(WorkflowPhase.PLANNING)
    execute = type(controller).execute
    attempts = []
    async def failing(self, context):
        attempts.append(True)
        if len(attempts) <= 2:
            raise RuntimeError("private-backend-detail")
        return await execute(self, context)
    monkeypatch.setattr(type(controller), "execute", failing)
    inputs = iter(("First turn.", "Please remember that I prefer concise explanations.", "/quit"))
    monkeypatch.setattr(cli, "get_settings", _settings)
    monkeypatch.setattr(cli, "DemoSession", lambda *a, **kw: session)
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    assert cli.main(["chat", "--live-trace", "--no-color"]) == 0
    output = capsys.readouterr().out
    assert "[!!] Planning - failed" in output and "PRIMA request failed" in output
    assert "private-backend-detail" not in output and "Traceback" not in output
    second = output.split("PRIMA EXECUTION")[2]
    assert "[!!]" not in second and "PRIMA > I will keep" in second
    assert not _observers(_bus(session))


@pytest.mark.parametrize("ending", ["quit", "eof", "interrupt", "request_interrupt"])
def test_cli_shutdown_detaches_observer(monkeypatch, capsys, ending):
    session, _ = _session(["--live-trace"])
    def read(prompt):
        if ending == "eof":
            raise EOFError
        if ending == "interrupt":
            raise KeyboardInterrupt
        return "turn" if ending == "request_interrupt" else "/quit"
    if ending == "request_interrupt":
        async def interrupted(request):
            raise KeyboardInterrupt
        monkeypatch.setattr(session.runtime, "execute", interrupted)
    monkeypatch.setattr(cli, "get_settings", _settings)
    monkeypatch.setattr(cli, "DemoSession", lambda *a, **kw: session)
    monkeypatch.setattr("builtins.input", read)
    assert cli.main(["chat", "--live-trace"]) == 0
    assert session._live_observer.closed and not _observers(_bus(session))
    output = capsys.readouterr().out
    assert "Traceback" not in output and "[!!]" not in output


def test_rendering_failure_does_not_fail_real_execution(monkeypatch):
    session, _ = _session(["--live-trace"])
    renderer = RecordingRenderer()
    session.observe_workflow(renderer)
    _bus(session).raise_subscriber_errors = True
    def broken(*args):
        raise RuntimeError("private-renderer-detail")
    monkeypatch.setattr(renderer, "live_event", broken)
    async def run():
        try:
            response = await session.send("Please remember my preference.")
            assert response.status is ExecutionStatus.COMPLETED
        finally:
            await session.close()
    asyncio.run(run())
    output = renderer.output.getvalue()
    assert output.count("Live workflow trace unavailable") == 1 and "private" not in output


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("live", [False, True])
def test_live_persistence_and_observability_are_independent(tmp_path, persist, live):
    session = open_session(tmp_path, persist=persist)
    session.live_trace = live
    renderer = RecordingRenderer()
    session.observe_workflow(renderer)
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handle = Tracing(provider, status="enabled")
    handle.client = SimpleNamespace(shutdown=lambda: None)
    session.observability = session.runtime.observability = handle
    async def run():
        try:
            response = await session.send("Please remember that I prefer concise explanations.")
            assert response.status is ExecutionStatus.COMPLETED
            assert session.runtime.state_manager.load(session.session_id).version == 1
        finally:
            await session.close()
        if persist:
            reopened = open_session(tmp_path)
            reopened.live_trace = live
            reopened.observe_workflow(renderer)
            try:
                assert reopened.runtime.state_manager.load(reopened.session_id).version == 1
                assert reopened.runtime.memory_repository.count() > 0
                await reopened.send("What kind of explanations do I prefer?")
                assert reopened.runtime.state_manager.load(reopened.session_id).version == 2
            finally:
                await reopened.close()
    try:
        asyncio.run(run())
        spans = exporter.get_finished_spans()
        assert sum(s.name == "prima.turn" for s in spans) == 1
        assert sum(s.name == "llm.generation" for s in spans) == 1
        assert len(session.runtime.llm_client.provider.requests) == 1
        assert ("PRIMA EXECUTION" in renderer.output.getvalue()) is live
    finally:
        handle.shutdown()
        provider.shutdown()


def test_preflight_does_not_subscribe_or_execute_workflow(monkeypatch, capsys):
    def forbidden(*args):
        pytest.fail("Preflight must not run live workflow observation")
    monkeypatch.setattr(LiveTraceObserver, "__init__", forbidden)
    monkeypatch.setattr(cli, "get_settings", _settings)
    assert cli.main(["preflight"]) == 0
    assert "PRIMA EXECUTION" not in capsys.readouterr().out
