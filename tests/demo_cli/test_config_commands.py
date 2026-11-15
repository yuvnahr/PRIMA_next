from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from io import StringIO

import pytest

from config.settings import Settings
from demo_cli.cli import build_parser
from demo_cli.commands import dispatch_command
from demo_cli.config import TraceMode, resolve_config
from demo_cli.renderer import Renderer
from demo_cli.session import DemoSession
from runtime import ExecutionProfile


@pytest.fixture
def settings(monkeypatch):
    for name in ("DEFAULT_PROVIDER", "DEFAULT_MODEL", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return Settings(_env_file=None)


def _config(argv, settings, env=None):
    return resolve_config(build_parser().parse_args(argv), settings, {} if env is None else env)


def test_cli_overrides_project_and_demo_environment(settings) -> None:
    configured = Settings(_env_file=None, default_provider="ollama", default_model="configured-local")
    config = _config([
        "chat", "--provider", "groq", "--model", "explicit-model", "--profile", "rag",
        "--trace", "off", "--session", "explicit-session", "--temperature", "0.4",
        "--max-tokens", "700", "--no-color",
    ], configured, {
        "PRIMA_DEMO_PROFILE": "model", "PRIMA_DEMO_TRACE": "full", "PRIMA_DEMO_SESSION": "environment",
        "PRIMA_DEMO_TEMPERATURE": "0.2", "PRIMA_DEMO_MAX_TOKENS": "800",
    })
    assert config.generation.provider == "groq" and config.generation.model == "explicit-model"
    assert config.profile is ExecutionProfile.SIMPLE_RAG and config.trace_mode is TraceMode.OFF
    assert config.session_id == "explicit-session" and config.no_color
    assert config.generation.temperature == 0.4 and config.generation.max_output_tokens == 700
    with pytest.raises(FrozenInstanceError):
        config.generation.model = "changed"


def test_environment_provider_model_and_demo_defaults(monkeypatch, settings) -> None:
    monkeypatch.setenv("DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("DEFAULT_MODEL", "local:exact")
    config = _config(["chat"], Settings(_env_file=None), {
        "PRIMA_DEMO_PROFILE": "model", "PRIMA_DEMO_TRACE": "full", "PRIMA_DEMO_SESSION": "from-env",
        "PRIMA_DEMO_TEMPERATURE": "0.2", "PRIMA_DEMO_MAX_TOKENS": "600", "NO_COLOR": "",
    })
    assert config.generation.provider == "ollama" and config.generation.model == "local:exact"
    assert config.profile is ExecutionProfile.MODEL_ONLY and config.trace_mode is TraceMode.FULL
    assert config.session_id == "from-env" and config.no_color
    assert config.generation.temperature == 0.2 and config.generation.max_output_tokens == 600


@pytest.mark.parametrize("alias,profile", [
    ("prima", ExecutionProfile.PRIMA_FULL), ("rag", ExecutionProfile.SIMPLE_RAG), ("model", ExecutionProfile.MODEL_ONLY),
])
def test_profile_mapping(settings, alias, profile) -> None:
    assert _config(["chat", "--profile", alias], settings).profile is profile


def test_groq_default_and_generated_session_ids(settings) -> None:
    first = _config(["chat", "--provider", "groq"], settings)
    second = _config(["chat", "--provider", "groq"], settings)
    assert first.generation.model == "openai/gpt-oss-120b"
    assert first.generation.max_output_tokens == 4096
    assert first.profile is ExecutionProfile.PRIMA_FULL and first.trace_mode is TraceMode.SUMMARY
    assert first.session_id.startswith("demo-") and first.session_id != second.session_id
    assert _config(["status"], settings).generation.model == settings.default_model


def test_ollama_model_must_be_explicit(settings) -> None:
    with pytest.raises(ValueError, match="Ollama requires --model"):
        _config(["chat", "--provider", "ollama"], settings)
    assert _config(["chat", "--provider", "ollama", "--model", "exact:tag"], settings).generation.model == "exact:tag"
    configured = Settings(_env_file=None, default_provider="ollama", default_model="configured:tag")
    assert _config(["chat"], configured).generation.model == "configured:tag"
    with pytest.raises(ValueError, match="Ollama requires --model"):
        _config(["chat", "--provider", "ollama", "--model", " "], settings)
    with pytest.raises(ValueError, match="Ollama requires --model"):
        _config(["chat"], Settings(_env_file=None, default_provider="ollama", default_model=" "))


@pytest.mark.parametrize("env,match", [
    ({"PRIMA_DEMO_PROFILE": "bad"}, "Profile"), ({"PRIMA_DEMO_TRACE": "bad"}, "Trace"),
    ({"PRIMA_DEMO_SESSION": " "}, "Session"), ({"PRIMA_DEMO_MAX_TOKENS": "bad"}, "integer"),
    ({"PRIMA_DEMO_MAX_TOKENS": "0"}, "positive"), ({"PRIMA_DEMO_TEMPERATURE": "3"}, "temperature"),
])
def test_invalid_configuration_fails_cleanly(settings, env, match) -> None:
    with pytest.raises(ValueError, match=match):
        _config(["chat"], settings, env)


def test_phase_one_command_dispatch_and_no_provider_switching(settings) -> None:
    config = _config(["chat", "--provider", "ollama", "--model", "exact:tag", "--session", "commands"], settings)
    session = DemoSession(config, settings=settings)
    output = StringIO()
    renderer = Renderer(no_color=True, output=output)
    async def run():
        for command in ("/help", "/status", "/trace", "/trace off", "/trace summary", "/trace full", "/trace invalid", "/trace full extra", "/profile model", "/state", "/unknown"):
            assert await dispatch_command(command, session, renderer)
        assert not await dispatch_command("/quit", session, renderer)
        assert not await dispatch_command("/exit", session, renderer)

    asyncio.run(run())
    assert session.trace_mode is TraceMode.FULL
    assert session.config is config and session.config.generation.model == "exact:tag"
    assert session.latest_response is None
    text = output.getvalue()
    assert "COMMANDS" in text and "PRIMA STATUS" in text and "commands" in text
    assert "Usage: /trace" in text and "Unknown command" in text
    assert "\x1b" not in text
