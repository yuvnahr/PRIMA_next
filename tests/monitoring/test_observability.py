from __future__ import annotations

import asyncio
import builtins
import logging
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from config.settings import Settings
from demo_cli import cli
from demo_cli.config import resolve_config
from llm.llm_types import LLMResponse
from monitoring import tracing
from runtime import ExecutionProfile, ExecutionStatus, PrimaRequest, PrimaRuntime, TaskKind
from tests.demo_cli.test_session_cli import _session
from uncertainty import UncertaintyGatePolicy


@pytest.fixture(autouse=True)
def disable_live_export(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setattr("config.settings._SETTINGS", Settings(_env_file=None, langfuse_enabled=False))


@pytest.fixture
def local_traces():
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    provider.shutdown()


def attach(session, provider):
    handle = tracing.Tracing(provider)
    session.observability = session.runtime.observability = handle
    return handle


def turn_spans(exporter):
    spans = exporter.get_finished_spans()
    roots = sorted((s for s in spans if s.name == "prima.turn"), key=lambda s: s.start_time)
    return [(root, {s.name: s for s in spans if s.context.trace_id == root.context.trace_id}) for root in roots]


def test_real_runtime_hierarchy_metadata_profiles_and_reset(local_traces):
    provider, exporter = local_traces
    session, fake = _session()
    handle = attach(session, provider)
    responses = []
    old_session = session.session_id

    async def run():
        await session.start()
        try:
            for profile in ("prima", "model", "rag"):
                session.set_profile(profile)
                responses.append(await session.send("Please remember that I prefer concise explanations."))
            await session.reset()
            assert session.runtime.observability is handle and session.runtime.llm_client.provider is fake
            responses.append(await session.send("A fresh conversation."))
        finally:
            await session.close()

    asyncio.run(run())
    turns = turn_spans(exporter)
    assert len(turns) == 4 and len({root.context.trace_id for root, _ in turns}) == 4
    for index, (root, children) in enumerate(turns):
        assert root.parent is None
        assert root.attributes["prima.session_id"] == (old_session if index < 3 else session.session_id)
        assert root.attributes["prima.execution_profile"] == responses[index].profile.value
        assert root.attributes["prima.provider"] == "groq"
        assert root.attributes["prima.model"] == session.config.generation.model
        assert root.attributes["prima.state_version_after"] == responses[index].diagnostics.state_version
        assert root.attributes["prima.latency_ms"] >= 0 and root.attributes["prima.status"] == "completed"
        assert root.attributes["prima.execution_id"]
        assert children["llm.generation"].parent.span_id == children["answer_generation"].context.span_id
        for name, child in children.items():
            if name not in {"prima.turn", "llm.generation", "memory_retrieval"}:
                assert child.parent.span_id == root.context.span_id
        if index == 0:
            response = responses[0]
            assert children["planning"].attributes["prima.action_count"] == response.output_data["plan"]["action_count"]
            assert children["planning"].attributes["prima.plan_confidence"] == response.output_data["plan"]["confidence"]
            assert children["world_simulation"].attributes["prima.risk_score"] == response.output_data["world_prediction"]["risk_score"]
            assert children["uncertainty_estimation"].attributes["prima.uncertainty_score"] == response.output_data["uncertainty"]["uncertainty"]
            assert children["execution_decision"].attributes["prima.execution_decision"] == response.output_data["execution_decision"]["decision"]
            assert children["affect"].attributes["prima.salience"] >= 0
            assert children["memory_retrieval"].parent.span_id == children["evidence_acquisition"].context.span_id
        else:
            assert not {"planning", "affect", "world_simulation", "uncertainty_estimation", "reflection"} & children.keys()
            assert ("memory_retrieval" in children) == (index != 1)
    serialized = str([(dict(s.attributes), s.events) for s in exporter.get_finished_spans()])
    assert "concise explanations" not in serialized and "fixture-key" not in serialized
    assert "I will keep your preference" not in serialized and "embedding" not in serialized


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 8, "completion_tokens": 3}, {"total_tokens": 11}])
def test_generation_records_only_reported_usage(local_traces, monkeypatch, usage):
    provider, exporter = local_traces
    session, fake = _session(["--profile", "model"])
    attach(session, provider)
    monkeypatch.setattr(fake, "send", lambda request: LLMResponse("hello", provider="groq", usage=usage))
    asyncio.run(session.send("private input"))
    generation = turn_spans(exporter)[0][1]["llm.generation"]
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert generation.attributes.get(f"prima.{key}") == (usage or {}).get(key)
    assert generation.attributes["prima.status"] == "completed"
    assert generation.attributes["prima.max_tokens"] == session.config.generation.max_output_tokens


def test_real_reflection_result_is_recorded_only_when_routed(local_traces, monkeypatch):
    provider, exporter = local_traces
    session, _ = _session()
    session.runtime = PrimaRuntime(
        llm_client=session.runtime.llm_client, generation_config=session.config.generation,
        runtime_config=session.runtime.runtime_config,
        uncertainty_policy=UncertaintyGatePolicy(reflect_uncertainty=0.0, max_retrieval_retries=0),
    )
    attach(session, provider)
    results = []
    evaluate = session.runtime.reflection_engine.evaluate

    def capture(context):
        result = evaluate(context)
        results.append(result)
        return result

    monkeypatch.setattr(session.runtime.reflection_engine, "evaluate", capture)
    asyncio.run(session.send("Please remember that I prefer concise explanations."))
    reflection = turn_spans(exporter)[0][1]["reflection"]
    assert reflection.attributes["prima.triggered"] == results[-1].should_reflect
    assert reflection.attributes["prima.trigger_score"] == results[-1].trigger_score
    assert reflection.attributes["prima.attempt"] == 1


def test_runtime_failure_has_safe_error_status_and_stage(local_traces, monkeypatch):
    provider, exporter = local_traces
    session, _ = _session()
    attach(session, provider)

    def failed_affect(*args, **kwargs):
        raise ValueError("Authorization: fixture-key")

    monkeypatch.setattr(session.runtime.affect_engine, "process", failed_affect)
    response = asyncio.run(session.send("private input"))
    root, children = turn_spans(exporter)[0]
    assert response.status is ExecutionStatus.FAILED and root.status.status_code is trace.StatusCode.ERROR
    assert root.attributes["prima.failed_stage"] == "affect"
    assert "prima.state_version_after" not in root.attributes
    assert children["affect"].attributes["prima.error_type"] == "ValueError"
    assert all(not child.events for child in children.values())
    assert "fixture-key" not in str([dict(child.attributes) for child in children.values()])


def test_disabled_never_imports_langfuse_and_retains_existing_provider(local_traces, monkeypatch):
    provider, exporter = local_traces
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "langfuse":
            pytest.fail("Disabled tracing must not import Langfuse")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    handle = tracing.configure_tracing(Settings(_env_file=None, langfuse_enabled=False), provider=provider)
    assert handle.status == "disabled" and handle.client is None
    with handle.span("existing"):
        pass
    assert exporter.get_finished_spans()[0].name == "existing"


@pytest.mark.parametrize("fields", [{}, {"langfuse_base_url": "https://["}, {"langfuse_base_url": "https://user:secret@host"}])
def test_missing_or_invalid_configuration_is_safe(local_traces, caplog, fields):
    provider, _ = local_traces
    settings = Settings(_env_file=None, langfuse_enabled=True, langfuse_public_key="private-public-key", **fields)
    result = tracing.configure_tracing(settings, provider=provider)
    assert result.status == "misconfigured" and result.client is None
    assert "private-public-key" not in caplog.text and "secret@host" not in caplog.text


def configured_settings():
    return Settings(_env_file=None, langfuse_enabled=True, langfuse_public_key="test-public", langfuse_secret_key="test-secret",  # noqa: S106 - synthetic fixture
                    langfuse_base_url="https://langfuse.invalid")


def test_sdk_initialization_failure_remains_noncritical(local_traces, monkeypatch, caplog):
    provider, _ = local_traces

    def broken(**kwargs):
        raise RuntimeError("Authorization: test-secret")

    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(Langfuse=broken, LangfuseGeneration=object,
                                                               propagate_attributes=object))
    result = tracing.configure_tracing(configured_settings(), provider=provider)
    assert result.status == "misconfigured" and result.provider is provider
    assert "test-secret" not in caplog.text


def test_console_initialization_is_idempotent_with_existing_provider(local_traces, monkeypatch):
    provider, existing = local_traces
    console = InMemorySpanExporter()
    monkeypatch.setattr(tracing, "ConsoleSpanExporter", lambda: console)
    global_provider = trace.get_tracer_provider()
    assert tracing.init_tracing(provider=provider) is provider
    assert tracing.init_tracing(provider=provider) is provider
    with provider.get_tracer("test").start_as_current_span("existing"):
        pass
    assert len(existing.get_finished_spans()) == len(console.get_finished_spans()) == 1
    assert trace.get_tracer_provider() is global_provider


def test_shared_provider_idempotent_initialization_and_filter(local_traces, monkeypatch):
    provider, existing = local_traces
    exported = InMemorySpanExporter()
    calls = []

    class FilteredProcessor(SpanProcessor):
        def on_end(self, observation):
            if calls[0]["should_export_span"](observation):
                exported.export((observation,))

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            kwargs["tracer_provider"].add_span_processor(FilteredProcessor())

        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(Langfuse=Client, LangfuseGeneration=object,
                                                               propagate_attributes=lambda **kwargs: None))
    monkeypatch.setattr(tracing, "_CLIENTS", {})
    global_provider = trace.get_tracer_provider()
    handle = tracing.configure_tracing(configured_settings(), provider=provider)
    assert tracing.configure_tracing(configured_settings(), provider=provider) is handle
    assert handle.status == "enabled" and len(calls) == 1
    assert trace.get_tracer_provider() is global_provider and handle.provider is provider
    with handle.turn(PrimaRequest(task_kind=TaskKind.CONVERSATION, profile=ExecutionProfile.MODEL_ONLY, input_text="private")):
        with provider.get_tracer("unrelated").start_as_current_span("noise"):
            pass
    assert len(existing.get_finished_spans()) == 2
    assert [s.name for s in exported.get_finished_spans()] == ["prima.turn"]
    handle.shutdown()
    handle.shutdown()
    assert calls.count("shutdown") == 1


@pytest.mark.parametrize("boundary", ["on_start", "on_end"])
def test_broken_processor_never_changes_runtime_answer(local_traces, boundary, caplog):
    provider, _ = local_traces

    class BrokenProcessor(SpanProcessor):
        def on_start(self, observation, parent_context=None):
            if boundary == "on_start":
                raise RuntimeError("Authorization: test-secret")

        def on_end(self, observation):
            if boundary == "on_end":
                raise RuntimeError("Authorization: test-secret")

    provider.add_span_processor(BrokenProcessor())
    session, _ = _session(["--profile", "model"])
    attach(session, provider)
    response = asyncio.run(session.send("hello"))
    assert response.status is ExecutionStatus.COMPLETED and response.output_text
    assert session.latest_diagnostics and "Observability span" in caplog.text
    assert "test-secret" not in caplog.text


def test_async_export_connectivity_failure_is_noncritical(local_traces, caplog):
    provider, _ = local_traces
    called = threading.Event()

    class FailingExporter(InMemorySpanExporter):
        def export(self, spans):
            called.set()
            logging.getLogger("langfuse").error("Authorization: test-secret")
            return SpanExportResult.FAILURE

    processor = BatchSpanProcessor(FailingExporter(), schedule_delay_millis=10)
    provider.add_span_processor(processor)
    logging.getLogger("langfuse").addFilter(tracing._ExportLogFilter())
    session, _ = _session(["--profile", "model"])
    attach(session, provider)
    response = asyncio.run(session.send("hello"))
    assert response.status is ExecutionStatus.COMPLETED and session.latest_diagnostics
    assert processor.force_flush(timeout_millis=500) and called.is_set()
    assert "Observability SDK/export diagnostic" in caplog.text and "test-secret" not in caplog.text


@pytest.mark.parametrize("failure", [False, True])
def test_shutdown_is_bounded_idempotent_and_noncritical(failure, caplog):
    handle = tracing.Tracing()
    release = threading.Event()
    calls = []

    def close():
        calls.append(True)
        if failure:
            raise RuntimeError("test-secret")
        release.wait(1)

    handle.client = SimpleNamespace(shutdown=close)
    started = time.monotonic()
    handle.shutdown(timeout=0.02)
    handle.shutdown(timeout=0.02)
    assert time.monotonic() - started < 0.5 and len(calls) == 1
    release.set()
    assert "test-secret" not in caplog.text


@pytest.mark.parametrize("ending", ["quit", "eof", "interrupt"])
def test_cli_exit_flushes_once_and_flag_overrides_default(monkeypatch, ending, capsys):
    session, _ = _session(["--profile", "model"])
    settings = Settings(_env_file=None, langfuse_enabled=False)
    args = cli.build_parser().parse_args(["chat", "--langfuse"])
    assert resolve_config(args, settings, {}).langfuse
    calls = []

    def read_input(prompt):
        if ending == "quit":
            return "/quit"
        raise EOFError if ending == "eof" else KeyboardInterrupt

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "DemoSession", lambda *args, **kwargs: session)
    monkeypatch.setattr(cli, "shutdown_tracing", lambda: calls.append(True))
    monkeypatch.setattr("builtins.input", read_input)
    assert cli.main(["chat", "--langfuse", "--no-color"]) == 0
    assert calls == [True] and "Langfuse" in capsys.readouterr().out


def test_official_langfuse_sdk_propagates_sessions_and_generation(local_traces, monkeypatch):
    langfuse = pytest.importorskip("langfuse")

    if not hasattr(langfuse, "propagate_attributes"):
        pytest.skip("Install requirements-observability.txt to exercise the modern optional SDK")
    provider, _ = local_traces
    exporter = InMemorySpanExporter()
    actual_client = langfuse.Langfuse
    monkeypatch.setattr(langfuse, "Langfuse", lambda **kwargs: actual_client(**kwargs, span_exporter=exporter))
    monkeypatch.setattr(tracing, "_CLIENTS", {})
    handle = tracing.configure_tracing(configured_settings(), provider=provider)
    assert handle.status == "enabled"
    session, _ = _session()
    session.observability = session.runtime.observability = handle
    previous = session.session_id

    async def run():
        await session.start()
        try:
            for alias in ("prima", "model", "rag"):
                session.set_profile(alias)
                await session.send("Please remember that I prefer concise explanations.")
            await session.reset()
            await session.send("A fresh conversation.")
        finally:
            await session.close()

    try:
        asyncio.run(run())
        handle.shutdown()
        turns = turn_spans(exporter)
        assert len(turns) == 4
        for index, (_root, children) in enumerate(turns):
            expected = previous if index < 3 else session.session_id
            assert all(child.attributes["session.id"] == expected for child in children.values())
            generation = children["llm.generation"]
            assert generation.attributes["langfuse.observation.type"] == "generation"
            assert generation.attributes["langfuse.observation.model.name"] == session.config.generation.model
            assert generation.attributes["langfuse.observation.usage_details"]
        assert "test-secret" not in str([dict(s.attributes) for s in exporter.get_finished_spans()])
    finally:
        handle.shutdown()
