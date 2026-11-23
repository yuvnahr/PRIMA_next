from __future__ import annotations

import asyncio
from dataclasses import replace
from io import StringIO

import pytest

from config.runtime_config import RuntimeConfig
from demo_cli import cli
from demo_cli.comparison import HISTORY_CHARACTER_LIMIT, ComparisonSession
from demo_cli.config import resolve_config
from demo_cli.renderer import Renderer
from llm.llm_client import LLMClient
from llm.llm_types import LLMResponse
from llm.prompt_builder import CONVERSATION_POLICY
from llm.provider import ProviderError
from monitoring.tracing import Tracing
from runtime import PrimaRuntime
from tests.demo_cli.test_session_cli import FakeProvider, _settings


def paired(monkeypatch, *, observability=None, flags=()):
    settings = _settings()
    monkeypatch.setattr("config.settings._SETTINGS", settings)
    args = cli.build_parser().parse_args(["compare", "--provider", "groq", "--no-color", *flags])
    config = resolve_config(args, settings, {})
    provider = FakeProvider(settings)
    provider.supports_history = True
    client = LLMClient(provider_name="groq", settings=settings, provider=provider)

    def send(request):
        provider.requests.append(request)
        prefix = "BASE" if "<UNTRUSTED_USER_INPUT>" not in request.prompt else "PRIMA"
        return LLMResponse(f"{prefix} answer {len(provider.requests)}", provider="groq",
                           usage={"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11})

    monkeypatch.setattr(provider, "send", send)
    return ComparisonSession(config, settings=settings, llm_client=client,
                             observability=observability or Tracing(), runtime_config=RuntimeConfig()), provider


def test_paired_conversations_have_independent_history_and_equal_settings(monkeypatch):
    session, provider = paired(monkeypatch)
    executions = []
    execute = PrimaRuntime.execute

    async def record(self, request):
        executions.append(request)
        return await execute(self, request)

    monkeypatch.setattr(PrimaRuntime, "execute", record)

    async def run():
        try:
            await session.prima.start()
            first = await session.send("Please remember I prefer concise explanations.")
            second = await session.send("What explanation style do I prefer?")
            assert first.base.text == "BASE answer 1" and first.prima.text == "PRIMA answer 2"
            assert first.base.calls == first.prima.calls == second.base.calls == second.prima.calls == 1
            assert first.base.usage == first.prima.usage == {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
            assert len(executions) == 2
            assert provider.requests[0].prompt == executions[0].input_text
            assert provider.requests[2].prompt == executions[1].input_text
            assert provider.requests[2].history == ({"role": "user", "content": executions[0].input_text},
                                                    {"role": "assistant", "content": first.base.text})
            assert first.prima.text not in str(provider.requests[2].history)
            assert not provider.requests[1].history and not provider.requests[3].history
            assert all(request.generation == session.config.generation for request in provider.requests)
            assert all(request.system_prompt == CONVERSATION_POLICY for request in provider.requests)
            assert session.base_session_id != session.prima.session_id
            assert session.prima.runtime.state_manager.load(session.base_session_id).version == 0
            assert all(note.context.get("source_session_id") == session.prima.session_id
                       for note in session.prima.runtime.memory_repository.list())
            assert session.prima.runtime.llm_client.call_count == 4
        finally:
            await session.close()

    asyncio.run(run())


def test_paired_reset_and_history_limit_never_silently_drop_context(monkeypatch):
    session, provider = paired(monkeypatch)

    async def run():
        try:
            await session.send("Please remember I prefer concise explanations.")
            previous = session.comparison_id, session.base_session_id, session.prima.session_id
            await session.reset()
            assert all(old != new for old, new in zip(previous, (session.comparison_id, session.base_session_id, session.prima.session_id), strict=True))
            assert not session.base_history and session.turn_index == 0 and session.latest_turn is None
            assert session.prima.runtime.memory_repository.count() == 0
            count = len(provider.requests)
            with pytest.raises(ValueError, match="neither branch ran"):
                await session.send("x" * HISTORY_CHARACTER_LIMIT)
            assert len(provider.requests) == count and session.turn_index == 0
            await session.send("Fresh request.")
            assert provider.requests[-2].history == () and session.turn_index == 1
        finally:
            await session.close()

    asyncio.run(run())


@pytest.mark.parametrize("branch", ["base", "prima"])
def test_partial_failure_keeps_other_answer_and_requires_reset(monkeypatch, branch):
    session, provider = paired(monkeypatch)
    send = provider.send

    def fail(request):
        is_prima = "<UNTRUSTED_USER_INPUT>" in request.prompt
        if is_prima == (branch == "prima"):
            provider.requests.append(request)
            raise ProviderError("HTTP request failed: HTTP 429")
        return send(request)

    monkeypatch.setattr(provider, "send", fail)

    async def run():
        try:
            turn = await session.send("A practical tip, please.")
            assert getattr(turn, branch).status == "failed"
            successful = turn.prima if branch == "base" else turn.base
            assert successful.status == "completed" and "answer" in successful.text
            assert turn.base.calls == turn.prima.calls == 1
            assert session.needs_reset
            with pytest.raises(ValueError, match="uneven histories"):
                await session.send("Another prompt")
            assert len(provider.requests) == 2
            await session.reset()
            assert not session.needs_reset
        finally:
            await session.close()

    asyncio.run(run())


@pytest.mark.parametrize("width", [38, 120])
def test_comparison_renders_complete_answers_at_both_widths(monkeypatch, width):
    session, _ = paired(monkeypatch)
    output = StringIO()

    async def run():
        try:
            turn = await session.send("Short prompt")
            long_base = "BASE-" * 180 + "END_BASE"
            long_prima = "PRIMA-" * 220 + "END_PRIMA"
            turn = replace(turn, base=replace(turn.base, text=long_base), prima=replace(turn.prima, text=long_prima))
            Renderer(output=output, no_color=True, width=width).comparison(turn)
            text = output.getvalue()
            assert "END_BASE" in text.replace("\n", "").replace(" ", "").replace("|", "")
            assert "END_PRIMA" in text.replace("\n", "").replace(" ", "").replace("|", "")
            answers, metrics = text.split("\n\nBASE METRICS\n", 1)
            base_metrics, prima_metrics = metrics.split("\n\nPRIMA METRICS\n", 1)
            for section in (base_metrics, prima_metrics):
                assert "End-to-end" in section
                assert "Tokens in/out/total 8/3/11" in " ".join(section.split())
            if width >= 100:
                heading, rule, *answer_lines = answers.split("\n\n", 1)[1].splitlines()
                assert "  |  " in heading and "  +  " in rule
                left, right = [], []
                for line in answer_lines:
                    a, b = line.split("  |  ", 1)
                    assert len(line) <= width
                    left.append(a.rstrip())
                    right.append(b)
                assert "".join(left) == long_base and "".join(right) == long_prima
            else:
                _, rendered_base, rendered_prima = answers.split("\n\n")
                rule = "-" * width
                assert rendered_base.splitlines()[:2] == ["BASE MODEL (ordinary chat)", rule]
                assert rendered_prima.splitlines()[:2] == ["PRIMA (same model + pipeline)", rule]
                assert "".join(rendered_base.splitlines()[2:]) == long_base
                assert "".join(rendered_prima.splitlines()[2:]) == long_prima
        finally:
            await session.close()

    asyncio.run(run())


def test_compare_cli_routes_commands_and_resets_both_branches(monkeypatch, capsys):
    session, _ = paired(monkeypatch)
    settings = _settings()
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr("demo_cli.comparison.ComparisonSession", lambda *args, **kwargs: session)
    inputs = iter(["A practical tip.", "/status", "/profile model", "/plan", "/affect", "/decision",
                   "/evidence", "/memory why", "/action", "/review", "/reset", "/affect", "/status", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    assert cli.main(["compare", "--provider", "groq", "--no-color"]) == 0
    text = capsys.readouterr().out
    assert "BASE MODEL (ordinary chat)" in text and "PRIMA (same model + pipeline)" in text
    assert "profiles are fixed" in text and "BOTH BRANCHES RESET" in text
    for title in ("AFFECT", "EXECUTION DECISION", "MEMORY EVIDENCE", "MEMORY ADMISSION", "ACTION RESULT", "TURN REVIEW"):
        assert f"\n{title}\n" in text
    assert "No turn response is available yet." in text
    assert session.turn_index == 0 and not session.base_history
    assert not session.prima.runtime.maintenance_supervisor.running


@pytest.mark.parametrize("flags,match", [(["--persist"], "process-local"), (["--profile", "model"], "--profile prima"),
                                         (["--trace-content"], "requires --langfuse")])
def test_compare_rejects_unsupported_or_unsafe_configuration(flags, match):
    args = cli.build_parser().parse_args(["compare", *flags])
    with pytest.raises(ValueError, match=match):
        resolve_config(args, _settings(), {})


def test_configuration_errors_reach_the_cli_without_becoming_generic(monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_settings", _settings)
    assert cli.main(["chat", "--trace-content"]) == 1
    assert "--trace-content requires --langfuse" in capsys.readouterr().out
    assert cli.main(["compare", "--persist"]) == 1
    assert "Comparison uses process-local histories" in capsys.readouterr().out
