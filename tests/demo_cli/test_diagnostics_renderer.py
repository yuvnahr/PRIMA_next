from __future__ import annotations

import re
from io import StringIO

import pytest

from demo_cli.config import TraceMode
from demo_cli.diagnostics import adapt_response
from demo_cli.renderer import PRIMA_BANNER, Renderer
from llm.generation_config import GenerationConfig
from runtime import (
    ExecutionOutcome,
    ExecutionProfile,
    ExecutionStatus,
    PrimaResponse,
    RuntimeComponent,
    RuntimeDiagnostics,
    TaskKind,
)


@pytest.mark.parametrize("mode", ["terminal", "pipe", "flag", "environment"])
@pytest.mark.parametrize("width", [20, 50, 80])
def test_welcome_shows_shadowed_logo_and_setup_frame(monkeypatch, mode, width):
    class Terminal(StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("WT_SESSION", "test-terminal")
    monkeypatch.delenv("NO_COLOR", raising=False)
    if mode == "environment":
        monkeypatch.setenv("NO_COLOR", "1")
    output = StringIO() if mode == "pipe" else Terminal()
    Renderer(output=output, width=width, no_color=mode == "flag").welcome(
        "Chat", (("Session", "q" * 100),),
        generation=GenerationConfig(model="fixture-model", provider="groq"), commands=(("/quit", "Exit"),),
    )
    rendered = output.getvalue()
    if mode == "terminal" and width > 20:
        assert "\x1b[38;2;57;91;179m" in rendered
        assert "\x1b[38;2;148;81;204m" in rendered
        assert "\x1b[48;2;57;91;179m" in rendered
        assert "\x1b[48;2;148;81;204m" in rendered
        assert "\x1b[48;2;24;35;72m" in rendered
    if mode == "terminal":
        assert "\x1b[38;2;57;91;179m|" in rendered
        assert "\x1b[38;2;148;81;204m|" in rendered
    else:
        assert "\x1b" not in rendered
    plain = re.sub(r"\x1b\[[0-9;]+m", "", rendered)
    logo_text, setup, menu = plain.split("\n\n", 2)
    logo = logo_text.splitlines()
    expected = PRIMA_BANNER.splitlines()
    if width == 20:
        expected = ["PRIMA"]
    elif width == 50:
        expected = [line[::2] for line in expected]
    assert [line.replace(".", " ").rstrip() for line in logo[:len(expected)]] == expected
    assert all(len(line) <= width for line in logo)
    if width > 20:
        offset = 2 if width == 80 else 1
        assert logo[-1] == " " * offset + expected[-1].replace("#", ".")
    frame = setup.splitlines()
    frame_width = min(width, 78)
    rail = "+" + "-" * (frame_width - 2) + "+"
    assert frame[0] == frame[2] == frame[-1] == rail
    assert all(len(line) == frame_width for line in frame)
    assert setup.count("q") == 100
    assert menu.startswith("COMMANDS\n/quit")


def _response(**kwargs):
    return PrimaResponse(
        request_id="turn-1", task_kind=TaskKind.CONVERSATION, profile=ExecutionProfile.PRIMA_FULL,
        status=ExecutionStatus.COMPLETED, outcome=ExecutionOutcome.ANSWERED, **kwargs,
    )


def test_adapter_uses_reported_values_and_excludes_private_payloads() -> None:
    response = _response(
        output_data={
            "affect_state": {"dominant_emotion": "fear", "valence": -0.7},
            "retrieval": {"candidate_count": 8, "selected_count": 3, "final_context_count": 2, "retrieval_confidence": 0.8},
            "plan": {"action_count": 5, "confidence": 0.9, "revision": 1},
            "world_prediction": {"risk_score": 0.28, "failure_probability": 0.22, "warnings": ["constraint pressure"]},
            "uncertainty": {"uncertainty_band": "low", "uncertainty": 0.2, "confidence": 0.8},
            "execution_decision": {"decision": "continue", "rule": "confidence_clearance"},
            "reflection_triggered": True, "memory_admission": {"stored": True},
            "private": {"raw_prompt": "secret body"},
        },
        diagnostics=RuntimeDiagnostics(
            state_version=2, latency_ms=812, provider={"name": "groq", "model": "target", "api_key": "secret key"},
            executed_components=(RuntimeComponent.AFFECT_ENGINE,), model_call_count=1, accepted_correction_count=1,
            trace_event_count=1, trace_events=({"event_type": "phase_completed", "phase": "affect", "status": "running", "payload": {"prompt": "secret payload"}},),
        ),
    )
    adapted = adapt_response(response)
    rows = dict(adapted.summary)
    assert rows["Affect"] == "fear / valence -0.700"
    assert rows["Retrieval"] == "3 selected / 8 candidates / 2 in context" and rows["Planning"] == "5 actions / confidence 0.900"
    assert rows["World model"] == "risk 0.280 / failure 0.220" and rows["Uncertainty"] == "low / 0.200"
    assert rows["Decision"] == "continue" and rows["Reflection"] == "triggered"
    assert rows["Commit"] == "memory admitted yes" and rows["State"] == "v2" and rows["Latency"] == "812 ms"
    assert dict(adapted.details)["Plan confidence"] == "0.900"
    assert "constraint pressure" in dict(adapted.details)["World warnings"]
    assert "secret" not in str(adapted)


@pytest.mark.parametrize("profile", [ExecutionProfile.MODEL_ONLY, ExecutionProfile.SIMPLE_RAG, ExecutionProfile.PRIMA_FULL])
def test_missing_and_skipped_modules_are_safe(profile) -> None:
    skipped = (RuntimeComponent.AFFECT_ENGINE, RuntimeComponent.PLANNER, RuntimeComponent.WORLD_MODEL) if profile is not ExecutionProfile.PRIMA_FULL else ()
    response = _response(diagnostics=RuntimeDiagnostics(skipped_components=skipped))
    response = response.model_copy(update={"profile": profile})
    adapted = adapt_response(response)
    rows = dict(adapted.summary)
    if profile is ExecutionProfile.MODEL_ONLY:
        assert "Planning" not in rows and rows["Generation"] == "n/a"
    else:
        assert rows["Planning"] == ("skipped" if skipped else "n/a")
        assert rows["Reflection"] == "n/a" and rows["Decision"] == "n/a"
    assert rows["State"] == rows["Latency"] == "n/a"
    response.output_data["world_prediction"] = {"risk_score": 0.0}
    assert dict(adapt_response(response).details)["World warnings"] == "n/a"


def test_full_trace_is_bounded_and_ignores_event_payloads() -> None:
    response = _response(
        output_data={"world_prediction": {"warnings": ["w" * 10000] * 1000}},
        diagnostics=RuntimeDiagnostics(
            trace_event_count=1000,
            trace_events=tuple({"event_type": "phase_completed", "phase": "answer_generation", "status": "running", "payload": {"raw": "private" * 10000}} for _ in range(1000)),
        ),
    )
    adapted = adapt_response(response)
    output = StringIO()
    renderer = Renderer(output=output, no_color=True, width=80)
    renderer.trace(adapted, TraceMode.FULL)
    text = output.getvalue()
    assert len(adapted.events) == 30 and "showing 30 of 1000" in text
    assert "private" not in text and len(text) < 12000
    assert "\x1b" not in text
    output.seek(0)
    output.truncate()
    renderer.trace(adapted, TraceMode.OFF)
    assert output.getvalue() == ""


def test_answer_wrapping_preserves_words_paragraphs_and_fenced_code() -> None:
    output = StringIO()
    renderer = Renderer(output=output, width=40, no_color=True)
    renderer.answer("The first thing I would focus on is completing the smallest useful task.\n\n```python\n    preserve_this_long_code_line_without_wrapping()\n```")
    text = output.getvalue()
    assert text.startswith("PRIMA > The first thing")
    assert " ".join(text.split()) == "PRIMA > The first thing I would focus on is completing the smallest useful task. ```python preserve_this_long_code_line_without_wrapping() ```"
    assert "\n        \n" in text
    assert "    preserve_this_long_code_line_without_wrapping()" in text
    assert all(len(line) <= 40 for line in text.splitlines()[:3])


def test_error_rendering_redacts_authorization_and_terminal_controls() -> None:
    output = StringIO()
    renderer = Renderer(output=output, no_color=True)
    renderer.error("HTTP 401 Authorization: Bearer fixture-key api_key=other-key \x1b[31mfailed", provider="groq", stage="answer_generation")
    text = output.getvalue()
    assert "HTTP 401" in text and "answer_generation" in text and "groq" in text
    assert "fixture-key" not in text and "other-key" not in text and "\x1b" not in text


def test_failed_stage_is_taken_from_runtime_event() -> None:
    response = _response(diagnostics=RuntimeDiagnostics(trace_events=({"event_type": "phase_failed", "phase": "answer_generation", "status": "failed"},)))
    assert adapt_response(response).failed_stage == "answer_generation"
