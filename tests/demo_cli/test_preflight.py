from __future__ import annotations

import builtins
import runpy
import sys
from types import SimpleNamespace

import pytest

from config.settings import Settings
from demo_cli import cli
from demo_cli.errors import safe_reason
from llm.llm_types import LLMResponse
from monitoring import tracing


@pytest.fixture
def configured(monkeypatch):
    settings = Settings(_env_file=None, default_provider="groq", groq_api_key="fixture-key",
                        langfuse_enabled=False, langfuse_public_key=None, langfuse_secret_key=None,
                        langfuse_base_url=None)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr("llm.provider.post_json", lambda *a, **kw: pytest.fail("No default preflight HTTP"))
    monkeypatch.setattr(tracing, "configure_tracing", lambda *a, **kw: pytest.fail("No preflight exporters"))
    return settings


def test_configured_groq_is_ready_without_request(configured, capsys):
    assert cli.main(["preflight"]) == 0
    output = capsys.readouterr().out
    for text in ("PRIMA DEMO PREFLIGHT", "READY", "groq", "openai/gpt-oss-120b", "prima_full", "in_memory", "disabled"):
        assert text in output
    assert "fixture-key" not in output and "Traceback" not in output


def test_missing_key_is_not_ready(configured, capsys):
    configured.groq_api_key = None
    assert cli.main(["preflight"]) == 1
    output = capsys.readouterr().out
    assert "missing" in output and "NOT READY" in output and "Set GROQ_API_KEY" in output


def test_ollama_explicit_model_is_configuration_only(configured, capsys):
    assert cli.main(["preflight", "--provider", "ollama", "--model", "exact:installed"]) == 0
    output = capsys.readouterr().out
    assert "exact:installed" in output and "not checked" in output and "READY" in output


def test_missing_provider(configured, capsys):
    configured.default_provider = " "
    assert cli.main(["preflight"]) == 1
    output = capsys.readouterr().out
    assert "NOT READY" in output and "Set DEFAULT_PROVIDER" in output


@pytest.mark.parametrize("complete", [False, True])
def test_optional_langfuse_config_and_missing_dependencies(configured, monkeypatch, capsys, complete):
    if complete:
        configured.langfuse_public_key = "private-public"
        configured.langfuse_secret_key = "private-secret"  # noqa: S105 - synthetic fixture
        configured.langfuse_base_url = "https://langfuse.invalid"
    monkeypatch.setitem(sys.modules, "langfuse", None)
    assert cli.main(["preflight", "--langfuse"]) == 0
    output = capsys.readouterr().out
    assert "READY WITH WARNING" in output and "optional" in output
    assert ("requirements-observability.txt" if complete else "LANGFUSE_PUBLIC_KEY") in output
    assert "private-public" not in output and "private-secret" not in output


def test_ready_observability_does_not_construct_exporter(configured, monkeypatch, capsys):
    configured.langfuse_enabled = True
    configured.langfuse_public_key = "private-public"
    configured.langfuse_secret_key = "private-secret"  # noqa: S105 - synthetic fixture
    configured.langfuse_base_url = "https://langfuse.invalid"
    def forbidden(**kwargs):
        pytest.fail("Preflight must not initialize Langfuse")
    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(
        Langfuse=forbidden, LangfuseGeneration=object, propagate_attributes=object,
    ))
    assert cli.main(["preflight"]) == 0
    output = capsys.readouterr().out
    assert "ready" in output and "WARNING" not in output


@pytest.mark.parametrize("failure", [False, True])
def test_strict_makes_one_tiny_provider_request(configured, monkeypatch, capsys, failure):
    calls = []
    def send(self, request):
        calls.append(request)
        if failure:
            raise RuntimeError("HTTP request failed: HTTP 401 private-secret")
        return LLMResponse("OK", provider="groq")
    monkeypatch.setattr("llm.provider.GroqProvider.send", send)
    assert cli.main(["preflight", "--strict"]) == int(failure)
    assert len(calls) == 1
    generation = calls[0].generation
    assert generation.max_output_tokens == 8 and generation.timeout_seconds == 10 and generation.retries == 0
    output = capsys.readouterr().out
    assert ("authentication failed" if failure else "Provider live") in output
    assert "private-secret" not in output and "Traceback" not in output


def test_runtime_construction_failure_is_safe(configured, monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise RuntimeError("private-secret")
    monkeypatch.setattr("demo_cli.preflight.DemoSession", broken)
    assert cli.main(["preflight"]) == 1
    output = capsys.readouterr().out
    assert "NOT READY" in output and "private-secret" not in output and "Traceback" not in output


@pytest.mark.parametrize("complete", [False, True])
def test_chat_continues_without_optional_langfuse(configured, monkeypatch, capsys, complete):
    if complete:
        configured.langfuse_public_key = "private-public"
        configured.langfuse_secret_key = "private-secret"  # noqa: S105 - synthetic fixture
        configured.langfuse_base_url = "https://langfuse.invalid"
    monkeypatch.setitem(sys.modules, "langfuse", None)
    monkeypatch.setattr("builtins.input", lambda prompt: "/quit")
    assert cli.main(["chat", "--provider", "ollama", "--model", "exact:installed", "--langfuse"]) == 0
    output = capsys.readouterr().out
    assert "PRIMA-NEXT Interactive Demo" in output and "Continuing without Langfuse" in output
    assert "private-secret" not in output and "Traceback" not in output


@pytest.mark.parametrize("error,provider,expected", [
    ("HTTP request failed: HTTP 401 private-secret", "groq", "authentication failed"),
    ("HTTP request failed: HTTP 503", "groq", "HTTP request failed"),
    ("HTTP request failed: ReadTimeout", "groq", "timed out"),
    ("HTTP request failed: ConnectionError", "ollama", "Ollama is unavailable"),
    ("HTTP request failed: HTTP 404", "ollama", "model was not found"),
    ("HTTP request failed: HTTP 429", "groq", "rate limit"),
    ("private-secret arbitrary failure", "groq", "operation did not complete"),
])
def test_presentation_errors_never_echo_raw_text(error, provider, expected):
    result = safe_reason(error, provider=provider)
    assert expected in result and "private-secret" not in result


def test_module_import_failure_has_no_traceback(monkeypatch, capsys):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "config.settings":
            raise ImportError("private-secret")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.delitem(sys.modules, "demo_cli.cli")
    with pytest.raises(SystemExit) as result:
        runpy.run_module("demo_cli.cli", run_name="__main__")
    assert result.value.code == 1
    output = capsys.readouterr().out
    assert "NOT READY" in output and "Traceback" not in output and "private-secret" not in output
