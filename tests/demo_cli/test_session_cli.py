from __future__ import annotations

import asyncio

import pytest

from config.runtime_config import RuntimeConfig
from config.settings import Settings
from demo_cli import cli
from demo_cli.config import TraceMode, resolve_config
from demo_cli.session import DemoSession
from llm.llm_client import LLMClient
from llm.llm_types import LLMResponse
from llm.provider import GroqProvider, Provider, ProviderError
from runtime import DiagnosticMode, ExecutionProfile, ExecutionStatus, PrimaRuntime, TaskKind


class FakeProvider(Provider):
    name = "groq"
    capabilities = GroqProvider.capabilities
    api_key = "fixture-key"

    def __init__(self, settings, *, fail_first=False):
        super().__init__(settings)
        self.requests = []
        self.fail_first = fail_first

    def send(self, request):
        self.requests.append(request)
        if self.fail_first and "<UNTRUSTED_USER_INPUT>\nfirst turn\n" in request.prompt:
            raise ProviderError("authentication failed; Authorization: Bearer fixture-key")
        return LLMResponse("I will keep your preference in mind.", provider="groq", usage={"total_tokens": 10})


def _settings():
    return Settings(_env_file=None, default_provider="groq", groq_api_key="fixture-key")


def _session(argv=None, fail_first=False):
    settings = _settings()
    args = cli.build_parser().parse_args(["chat", "--provider", "groq", "--session", "multi-turn", *(argv or [])])
    config = resolve_config(args, settings, {})
    provider = FakeProvider(settings, fail_first=fail_first)
    client = LLMClient(provider_name="groq", settings=settings, provider=provider)
    return DemoSession(config, settings=settings, llm_client=client, runtime_config=RuntimeConfig()), provider


def test_two_turns_use_same_real_runtime_session_and_memory(monkeypatch) -> None:
    requests = []
    execute = PrimaRuntime.execute

    async def recording_execute(self, request):
        requests.append((self, request))
        return await execute(self, request)

    monkeypatch.setattr(PrimaRuntime, "execute", recording_execute)
    session, provider = _session()
    runtime = session.runtime

    async def run():
        await session.start()
        try:
            first = await session.send("Please remember that I prefer concise explanations.")
            assert first.state_delta.changes["committed"]
            assert first.output_data["memory_ids_created"]
            session.trace_mode = TraceMode.OFF
            second = await session.send("What kind of explanations do I prefer?")
            assert second.status is ExecutionStatus.COMPLETED
            assert second.diagnostics.state_version == first.diagnostics.state_version + 1
            assert second.output_data["retrieval"]["final_context_count"] > 0
            assert any("concise explanations" in item.text for item in second.evidence)
            assert session.latest_response is second and session.latest_diagnostics is not None
            assert session.runtime is runtime and session.session_id == "multi-turn"
        finally:
            await session.close()

    asyncio.run(run())
    assert len(requests) == len(provider.requests) == 2
    assert all(owner is runtime for owner, _ in requests)
    assert all(request.session_id == "multi-turn" for _, request in requests)
    assert all(request.task_kind is TaskKind.CONVERSATION for _, request in requests)
    assert all(request.profile is ExecutionProfile.PRIMA_FULL for _, request in requests)
    assert all(request.diagnostic_mode is DiagnosticMode.DIAGNOSTIC for _, request in requests)
    assert not runtime.maintenance_supervisor.running
    assert runtime.maintenance_supervisor.diagnostics()["queue_size"] == 0


def test_cli_recovers_after_failure_and_commands_do_not_execute_turns(monkeypatch, capsys) -> None:
    sessions = []
    requests = []
    execute = PrimaRuntime.execute
    inputs = iter(("", "/help", "/status", "/trace full", "/trace invalid", "/trace off", "/unknown", "first turn", "second turn", "/quit"))

    async def recording_execute(self, request):
        requests.append(request)
        return await execute(self, request)

    def create_session(config, *, settings):
        provider = FakeProvider(settings, fail_first=True)
        session = DemoSession(
            config, settings=settings, llm_client=LLMClient(provider_name="groq", settings=settings, provider=provider),
            runtime_config=RuntimeConfig(),
        )
        sessions.append(session)
        return session

    monkeypatch.setattr(cli, "DemoSession", create_session)
    monkeypatch.setattr(PrimaRuntime, "execute", recording_execute)
    monkeypatch.setattr(cli, "get_settings", _settings)
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    assert cli.main(["chat", "--provider", "groq", "--profile", "model", "--no-color"]) == 0
    output = capsys.readouterr().out
    assert len(sessions) == 1 and len(requests) == 2
    assert "PRIMA request failed" in output and "PRIMA > I will keep your preference" in output
    assert "authentication failed" in " ".join(output.split()) and "fixture-key" not in output
    assert "COMMANDS" in output and "Usage: /trace" in output and "Unknown command" in output
    assert "PRIMA TRACE" not in output and "\x1b" not in output and "Traceback" not in output
    assert sessions[0].latest_response.status is ExecutionStatus.COMPLETED
    assert not sessions[0].runtime.maintenance_supervisor.running


@pytest.mark.parametrize("interrupt", [EOFError, KeyboardInterrupt])
def test_idle_exit_closes_maintenance(monkeypatch, capsys, interrupt) -> None:
    session, _ = _session(["--profile", "model"])

    def read_input(prompt):
        assert prompt == "You > "
        raise interrupt

    monkeypatch.setattr(cli, "get_settings", _settings)
    monkeypatch.setattr(cli, "DemoSession", lambda *args, **kwargs: session)
    monkeypatch.setattr("builtins.input", read_input)
    assert cli.main(["chat", "--provider", "groq", "--no-color"]) == 0
    assert not session.runtime.maintenance_supervisor.running
    assert "Traceback" not in capsys.readouterr().out


def test_missing_groq_key_fails_before_prompt_without_network(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None, groq_api_key=None))
    monkeypatch.setattr("llm.provider.post_json", lambda *args, **kwargs: pytest.fail("No HTTP during preflight"))
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Preflight must fail before prompt"))
    assert cli.main(["chat", "--provider", "groq", "--no-color"]) == 1
    assert "Set GROQ_API_KEY" in capsys.readouterr().out


def test_status_is_configuration_only_and_ollama_is_selectable(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr("llm.provider.post_json", lambda *args, **kwargs: pytest.fail("No HTTP during status"))
    assert cli.main(["status", "--provider", "ollama", "--model", "exact:installed", "--session", "status-session", "--no-color"]) == 0
    output = capsys.readouterr().out
    for value in ("PRIMA STATUS", "ollama", "exact:installed", "prima_full", "status-session", "in_memory", "summary"):
        assert value in output


def test_unknown_provider_and_settings_failure_are_safe(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "get_settings", _settings)
    assert cli.main(["status", "--provider", "unknown", "--no-color"]) == 1
    assert "Unknown provider" in capsys.readouterr().out

    def broken_settings():
        raise ValueError("fixture-key")

    monkeypatch.setattr(cli, "get_settings", broken_settings)
    assert cli.main(["status", "--no-color"]) == 1
    assert "fixture-key" not in capsys.readouterr().out
