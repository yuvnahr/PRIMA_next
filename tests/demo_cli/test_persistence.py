from __future__ import annotations

import asyncio
import hashlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from types import SimpleNamespace

import pytest

from config.runtime_config import RuntimeConfig
from config.settings import Settings
from demo_cli import cli
from demo_cli.commands import dispatch_command
from demo_cli.config import TraceMode, resolve_config
from demo_cli.renderer import Renderer
from demo_cli.session import DemoSession
from demo_cli.storage import PersistenceError, clear_session, display_data_root, session_directory
from llm.llm_client import LLMClient
from memory.memory_metadata import configure_collection_sync
from memory.memory_note import MemoryNote
from memory.memory_repository import ChromaMemoryRepository, InMemoryMemoryRepository
from memory.memory_types import MemoryType
from monitoring.tracing import Tracing
from runtime import ExecutionProfile, ExecutionStatus
from state.state_manager import JsonStateManager
from tests.demo_cli.test_session_cli import FakeProvider


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr("llm.provider.post_json", lambda *a, **kw: pytest.fail("No real provider requests"))


def config_for(root, session="alpha", *, persist=True, extra=(), settings=None, env=None):
    settings = settings or Settings(_env_file=None, default_provider="groq", groq_api_key="fixture-key", langfuse_enabled=False)
    args = cli.build_parser().parse_args([
        "chat", "--data-dir", str(root), "--session", session, *(('--persist',) if persist else ()), *extra,
    ])
    return resolve_config(args, settings, env or {}), settings


def open_session(root, session="alpha", *, persist=True, runtime_config=None, extra=()):
    config, settings = config_for(root, session, persist=persist, extra=extra)
    provider = FakeProvider(settings)
    client = LLMClient(provider_name="groq", settings=settings, provider=provider)
    return DemoSession(config, settings=settings, llm_client=client,
                       runtime_config=runtime_config or RuntimeConfig(), observability=Tracing())


def test_configuration_defaults_precedence_and_generated_identity(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, default_provider="groq", prima_demo_data_dir="from-project")
    parser = cli.build_parser()
    base = resolve_config(parser.parse_args(["chat"]), settings, {})
    assert not base.persist and base.data_dir.name == "from-project"
    assert base.session_id.startswith("demo-")
    configured = resolve_config(parser.parse_args(["chat", "--persist"]), settings,
                                {"PRIMA_DEMO_DATA_DIR": str(tmp_path / "env")})
    assert configured.persist and configured.data_dir == tmp_path / "env"
    explicit = resolve_config(parser.parse_args(["chat", "--persist", "--data-dir", str(tmp_path / "flag"),
                                                "--session", "explicit"]), settings,
                              {"PRIMA_DEMO_DATA_DIR": str(tmp_path / "env")})
    assert explicit.session_id == "explicit" and explicit.data_dir == tmp_path / "flag"
    monkeypatch.setenv("PRIMA_DEMO_DATA_DIR", str(tmp_path / "setting"))
    assert Settings(_env_file=None).prima_demo_data_dir == str(tmp_path / "setting")
    with pytest.raises(ValueError, match="directory"):
        resolve_config(parser.parse_args(["chat", "--data-dir", " "]), settings, {})


@pytest.mark.parametrize("session_id", ["../outside\\unsafe", "CAPSTONE:demo?", "नमस्ते", "a" * 500])
def test_namespace_is_safe_deterministic_and_collision_resistant(tmp_path, session_id):
    config, _ = config_for(tmp_path, session_id)
    directory = session_directory(config, session_id)
    assert directory == tmp_path / "sessions" / hashlib.sha256(session_id.encode()).hexdigest()
    assert session_directory(config, session_id) == directory
    assert session_directory(config, session_id + "x") != directory


def test_in_memory_stays_fresh_without_chroma_or_data_directory(tmp_path, monkeypatch):
    root = tmp_path / "unused"
    monkeypatch.setitem(sys.modules, "chromadb", None)
    first = open_session(root, persist=False)
    async def run():
        await first.start()
        await first.send("Please remember that I prefer concise explanations.")
        await first.close()
        second = open_session(root, persist=False)
        try:
            assert isinstance(second.runtime.memory_repository, InMemoryMemoryRepository)
            assert second.runtime.state_manager.load("alpha").version == 0
            assert not second.runtime.memory_repository.list()
            assert dict(second.status_rows())["Persistence"] == "disabled"
        finally:
            await second.close()
    asyncio.run(run())
    assert not root.exists()


def test_real_persistent_runtime_resumes_state_affect_memory_and_retrieval(tmp_path, monkeypatch):
    first = open_session(tmp_path)
    async def run():
        await first.start()
        await first.send("Please remember that I prefer concise explanations. I am nervous about my presentation.")
        await first.send("What kind of explanations do I prefer?")
        await first.runtime.flush_maintenance()
        before = first.runtime.state_manager.load("alpha")
        notes = {note.id: (note.content, note.embedding, note.to_metadata()) for note in first.runtime.memory_repository.list()}
        assert before.version == 2 and notes and before.emotional.emotional_momentum
        await first.close()
        with monkeypatch.context() as patch:
            patch.setattr("memory.embedding_pipeline.embed_text", lambda *a, **kw: pytest.fail("No re-embedding on reopen"))
            second = open_session(tmp_path)
            assert second.runtime.state_manager.load("alpha").to_dict() == before.to_dict()
            assert {note.id: (note.content, note.embedding, note.to_metadata()) for note in second.runtime.memory_repository.list()} == notes
            assert "concise explanations" in str(second.stored_memories())
            assert dict(second.current_state())["Version"] == "2"
            assert "Resumed persisted session: state version 2" in second.persistence_messages()[0]
            assert "No turn" in str(second.latest_plan()) and "No turn" in str(second.latest_world())
        seen = []
        process = second.runtime.affect_engine.process
        def recorded(text, **kwargs):
            seen.append(kwargs["cognitive_state"].emotional.to_dict())
            return process(text, **kwargs)
        monkeypatch.setattr(second.runtime.affect_engine, "process", recorded)
        try:
            await second.start()
            response = await second.send("What kind of explanations do I prefer?")
            assert response.status is ExecutionStatus.COMPLETED and response.diagnostics.state_version == 3
            assert seen == [before.emotional.to_dict()]
            assert any("concise explanations" in item.text for item in response.evidence)
            assert second.runtime.memory_index.graph_repository.client is second.runtime.memory_repository.client
        finally:
            await second.close()
    asyncio.run(run())


def test_sessions_isolate_state_affect_memories_and_retrieval(tmp_path):
    async def run():
        for label, preference in (("alpha", "concise explanations"), ("beta", "detailed explanations")):
            session = open_session(tmp_path, label)
            await session.start()
            await session.send(f"Please remember that I prefer {preference}.")
            await session.close()
        for label, wanted, unwanted in (("alpha", "concise", "detailed"), ("beta", "detailed", "concise")):
            session = open_session(tmp_path, label)
            try:
                assert session.runtime.state_manager.load(label).version == 1
                assert session.runtime.state_manager.load("foreign").version == 0
                assert session.runtime.state_manager.load("foreign").emotional.emotional_momentum == {}
                content = str(session.stored_memories())
                assert wanted in content and unwanted not in content
                await session.start()
                response = await session.send("What kind of explanations do I prefer?")
                assert response.evidence and all(unwanted not in item.text for item in response.evidence)
            finally:
                await session.close()
    asyncio.run(run())


def test_reset_clears_only_old_session_preserves_configuration_and_uses_new_namespace(tmp_path):
    async def run():
        session = open_session(tmp_path, "active")
        other = open_session(tmp_path, "other")
        await session.start()
        await other.start()
        await session.send("Please remember that I prefer concise explanations.")
        await session.send("What explanations do I prefer?")
        await other.send("Please remember that I prefer detailed explanations.")
        await other.runtime.flush_maintenance()
        untouched = other.runtime.state_manager.load("other").to_dict()
        other_notes = {note.id for note in other.runtime.memory_repository.list()}
        session.set_profile("rag")
        session.trace_mode = TraceMode.FULL
        old, client, telemetry = session.runtime, session.runtime.llm_client, session.observability
        output = StringIO()
        try:
            assert await dispatch_command("/reset", session, Renderer(output=output))
            assert session.session_id != "active" and not old.maintenance_supervisor.running
            assert old.memory_repository._closed
            assert session.runtime.llm_client is client and session.observability is telemetry
            assert session.current_profile is ExecutionProfile.SIMPLE_RAG and session.trace_mode is TraceMode.FULL
            assert session.config.persist and session.config.data_dir == tmp_path
            assert session.runtime.state_manager.load(session.session_id).version == 0
            assert session.runtime.memory_repository.count() == 0
            assert not session.runtime.affect_engine.history.get_recent(50)
            assert str(tmp_path / "sessions") in str(session.runtime.memory_repository.path)
            reopened = open_session(tmp_path, "active")
            try:
                assert reopened.runtime.state_manager.load("active").version == 0
                assert not reopened.runtime.memory_repository.list()
                assert not reopened.runtime.memory_index.graph_repository.nodes
            finally:
                await reopened.close()
            assert other.runtime.state_manager.load("other").to_dict() == untouched
            assert {note.id for note in other.runtime.memory_repository.list()} == other_notes
            await session.send("Please remember that I prefer slow explanations.")
            assert session.runtime.state_manager.load(session.session_id).version == 1
        finally:
            await session.close()
            await other.close()
        assert "SESSION RESET" in output.getvalue() and "To resume:" in output.getvalue()
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["memory", "state", "maintenance", "close", "runtime"])
def test_reset_failure_closes_session_without_claiming_success(tmp_path, monkeypatch, failure):
    session = open_session(tmp_path)
    async def run():
        await session.start()
        await session.send("Please remember that I prefer concise explanations.")
        def broken(*args, **kwargs):
            raise OSError("private-backend-detail")
        if failure in {"memory", "state"}:
            target = session.runtime.memory_index if failure == "memory" else session.runtime.state_manager
            monkeypatch.setattr(target, "clear" if failure == "memory" else "reset", broken)
        elif failure == "maintenance":
            stop = session.runtime.stop_maintenance
            async def failed_stop(**kwargs):
                await stop(**kwargs)
                broken()
            monkeypatch.setattr(session.runtime, "stop_maintenance", failed_stop)
        elif failure == "close":
            close = session.runtime.memory_index.close
            def failed_close():
                close()
                broken()
            monkeypatch.setattr(session.runtime.memory_index, "close", failed_close)
        else:
            monkeypatch.setattr(session, "_create_runtime", broken)
        with pytest.raises(PersistenceError, match="reset did not complete"):
            await session.reset()
        assert session.session_id == "alpha" and session._closed
        assert not session.runtime.maintenance_supervisor.running
        assert session.runtime.memory_repository._closed
        expected = 0 if failure in {"close", "runtime"} else 1
        assert JsonStateManager(session_directory(session.config, "alpha") / "state.json").load("alpha").version == expected
        with pytest.raises(PersistenceError, match="closed"):
            await session.send("Do not run this")
        with pytest.raises(PersistenceError, match="closed"):
            await session.reset()
        await session.close()
    asyncio.run(run())


def test_reset_refuses_redirected_backend(tmp_path):
    session = open_session(tmp_path)
    original_path = session.runtime.memory_repository.path
    try:
        session.runtime.memory_repository.path = tmp_path / "other"
        with pytest.raises(PersistenceError, match="outside the active session"):
            clear_session(session.config, session.session_id, session.runtime)
    finally:
        session.runtime.memory_repository.path = original_path
        asyncio.run(session.close())


@pytest.mark.parametrize("payload", ["not JSON", '{"schema_version":"unsupported","sessions":{}}'])
def test_corrupt_state_is_rejected_without_overwrite(tmp_path, payload):
    config, _ = config_for(tmp_path)
    directory = session_directory(config, "alpha")
    directory.mkdir(parents=True)
    path = directory / "state.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(PersistenceError, match="Cannot open"):
        open_session(tmp_path)
    assert path.read_text() == payload


@pytest.mark.parametrize("mismatch", ["dimension", "representation"])
def test_incompatible_embeddings_fail_safely_without_rebuild(tmp_path, mismatch):
    first = open_session(tmp_path)
    async def run():
        await first.start()
        await first.send("Please remember that I prefer concise explanations.")
        await first.close()
    asyncio.run(run())
    metadata = first.runtime.memory_repository.metadata_path.read_bytes()
    altered = RuntimeConfig(embedding_dimensions=32) if mismatch == "dimension" else RuntimeConfig(representation_mode="raw")
    with pytest.raises(PersistenceError, match="incompatible"):
        open_session(tmp_path, runtime_config=altered)
    assert first.runtime.memory_repository.metadata_path.read_bytes() == metadata
    reopened = open_session(tmp_path)
    try:
        assert reopened.runtime.memory_repository.count() > 0
    finally:
        asyncio.run(reopened.close())


def test_generation_model_does_not_bind_persisted_embeddings(tmp_path):
    first = open_session(tmp_path)
    async def run():
        await first.start()
        await first.send("Please remember that I prefer concise explanations.")
        await first.close()
    asyncio.run(run())
    second = open_session(tmp_path, extra=("--model", "another-generation-model"))
    assert second.config.generation.model == "another-generation-model"
    assert second.runtime.memory_repository.count() > 0
    asyncio.run(second.close())


def test_reset_old_namespace_can_start_with_a_new_embedding_dimension(tmp_path):
    first = open_session(tmp_path)
    async def run():
        await first.start()
        await first.send("Please remember that I prefer concise explanations.")
        await first.reset()
        await first.close()
        second = open_session(tmp_path, runtime_config=RuntimeConfig(embedding_dimensions=32))
        try:
            assert second.runtime.memory_repository.count() == 0
            await second.start()
            response = await second.send("Please remember that I prefer detailed explanations.")
            assert response.status is ExecutionStatus.COMPLETED
            assert all(len(note.embedding) == 32 for note in second.runtime.memory_repository.list())
        finally:
            await second.close()
        third = open_session(tmp_path, runtime_config=RuntimeConfig(embedding_dimensions=32))
        try:
            assert third.runtime.memory_repository.count() > 0
            assert all(len(note.embedding) == 32 for note in third.runtime.memory_repository.list())
        finally:
            await third.close()
    asyncio.run(run())


@pytest.mark.parametrize("payload,match", [("not JSON", "Cannot open"), ('{"unexpected":"metadata"}', "incompatible")])
def test_invalid_memory_metadata_is_rejected_without_overwrite(tmp_path, payload, match):
    config, _ = config_for(tmp_path)
    directory = session_directory(config, "alpha") / "chroma"
    directory.mkdir(parents=True)
    path = directory / ".prima_embedding_metadata.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(PersistenceError, match=match):
        open_session(tmp_path)
    assert path.read_text() == payload


def test_windows_cross_drive_data_root_is_displayable(tmp_path, monkeypatch):
    config, _ = config_for(tmp_path)
    def another_drive(*args):
        raise ValueError("path is on another mount")
    monkeypatch.setattr("demo_cli.storage.os.path.relpath", another_drive)
    assert display_data_root(config) == str(tmp_path)


@pytest.mark.parametrize("strict", [False, True])
def test_persistent_preflight_never_advances_state_or_inserts_memories(tmp_path, monkeypatch, capsys, strict):
    session = open_session(tmp_path)
    ids = set()
    async def run():
        await session.start()
        await session.send("Please remember that I prefer concise explanations.")
        await session.runtime.flush_maintenance()
        ids.update(note.id for note in session.runtime.memory_repository.list())
        await session.close()
    asyncio.run(run())
    state_path = session_directory(session.config, "alpha") / "state.json"
    state_before = state_path.read_bytes()
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None, default_provider="ollama", langfuse_enabled=False))
    monkeypatch.setattr("llm.provider.OllamaProvider.send", lambda self, request: FakeProvider(None).send(request))
    assert cli.main(["preflight", "--persist", "--data-dir", str(tmp_path), "--session", "alpha", "--model", "local:exact",
                     *(("--strict",) if strict else ())]) == 0
    assert state_path.read_bytes() == state_before
    reopened = open_session(tmp_path)
    try:
        assert {note.id for note in reopened.runtime.memory_repository.list()} == ids
    finally:
        asyncio.run(reopened.close())
    output = capsys.readouterr().out
    assert "READY" in output and "State store" in output and "Memory store" in output


def test_default_persistence_preflight_is_read_only_and_missing_chroma_affects_only_persistence(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None, default_provider="ollama", langfuse_enabled=False))
    root = tmp_path / "not-created"
    flags = ["--data-dir", str(root), "--model", "local:exact"]
    assert cli.main(["preflight", "--persist", *flags]) == 0
    assert not root.exists()
    monkeypatch.setitem(sys.modules, "chromadb", None)
    assert cli.main(["preflight", "--persist", *flags]) == 1
    assert cli.main(["preflight", *flags]) == 0
    output = capsys.readouterr().out
    assert "Persistence is unavailable" in output and "NOT READY" in output


def test_invalid_data_directory_is_presentation_safe(tmp_path, monkeypatch, capsys):
    path = tmp_path / "a-file"
    path.write_text("keep")
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None, default_provider="ollama", langfuse_enabled=False))
    assert cli.main(["preflight", "--persist", "--data-dir", str(path), "--model", "local:exact"]) == 1
    assert "Traceback" not in capsys.readouterr().out and path.read_text() == "keep"


@pytest.mark.parametrize("ending", ["quit", "eof", "interrupt", "reset"])
def test_persistent_cli_cleanup_is_idempotent_and_independent_of_telemetry(tmp_path, monkeypatch, capsys, ending):
    session = open_session(tmp_path)
    closed = []
    close = ChromaMemoryRepository.close
    def counted_close(self):
        closed.append(self)
        close(self)
    monkeypatch.setattr(ChromaMemoryRepository, "close", counted_close)
    inputs = iter(("/reset", "/quit") if ending == "reset" else ("/quit",))
    def read(prompt):
        if ending == "eof":
            raise EOFError
        if ending == "interrupt":
            raise KeyboardInterrupt
        return next(inputs)
    telemetry = []
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "DemoSession", lambda *a, **kw: session)
    monkeypatch.setattr("builtins.input", read)
    monkeypatch.setattr(cli, "shutdown_tracing", lambda: telemetry.append(True))
    assert cli.main(["chat", "--persist", "--provider", "ollama", "--model", "local:exact", "--data-dir", str(tmp_path)]) == 0
    asyncio.run(session.close())
    assert len(closed) == (2 if ending == "reset" else 1)
    assert len({id(repository) for repository in closed}) == len(closed)
    assert telemetry == [True] and "Traceback" not in capsys.readouterr().out


def test_generated_persistent_session_is_displayed_with_resume_command(tmp_path, monkeypatch, capsys):
    settings = Settings(_env_file=None, default_provider="ollama", default_model="local:exact", langfuse_enabled=False)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr("builtins.input", lambda prompt: "/quit")
    assert cli.main(["chat", "--persist", "--data-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "Started new persistent session" in output and "To resume:" in output and "--session 'demo-" in output
    assert "Persistence" in output and "enabled" in output


def test_failed_persistent_reset_exits_cli_without_success_or_traceback(tmp_path, monkeypatch, capsys):
    session = open_session(tmp_path)
    def broken():
        raise OSError("private-backend-detail")
    monkeypatch.setattr(session.runtime.memory_index, "clear", broken)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "DemoSession", lambda *a, **kw: session)
    monkeypatch.setattr("builtins.input", lambda prompt: "/reset")
    assert cli.main(["chat", "--persist", "--data-dir", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "reset did not complete" in output and "SESSION RESET" not in output
    assert "Traceback" not in output and "private-backend-detail" not in output


def test_persistent_repository_serializes_reads_with_maintenance_writes(tmp_path, monkeypatch):
    repository = ChromaMemoryRepository(str(tmp_path / "chroma"))
    entered, release, reading, read_entered = (threading.Event() for _ in range(4))
    collection = repository.collections[MemoryType.EPISODIC]
    add, get = collection.add, collection.get
    def gated_add(**kwargs):
        entered.set()
        assert release.wait(2)
        return add(**kwargs)
    def read():
        reading.set()
        return repository.list(MemoryType.EPISODIC)
    def checked_get(**kwargs):
        read_entered.set()
        assert release.is_set()
        return get(**kwargs)
    monkeypatch.setattr(collection, "add", gated_add)
    monkeypatch.setattr(collection, "get", checked_get)
    note = MemoryNote.create("Please remember that I prefer concise explanations.")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            write = pool.submit(repository.add, note)
            assert entered.wait(2)
            result = pool.submit(read)
            try:
                assert reading.wait(2)
                assert not read_entered.wait(0.03)
            finally:
                release.set()
            write.result(timeout=5)
            assert [stored.id for stored in result.result(timeout=5)] == [note.id]
            assert collection.configuration["hnsw"]["sync_threshold"] == 1
            assert list(repository.path.rglob("index_metadata.pickle"))
    finally:
        repository.close()


@pytest.mark.parametrize("modern", [False, True])
def test_collection_sync_uses_public_api_and_preserves_legacy_compatibility(modern):
    calls = []
    collection = SimpleNamespace(modify=lambda **kwargs: calls.append(kwargs))
    if modern:
        collection.configuration = {}
    configure_collection_sync(collection)
    assert calls == ([{"configuration": {"hnsw": {"sync_threshold": 1}}}] if modern else [])
