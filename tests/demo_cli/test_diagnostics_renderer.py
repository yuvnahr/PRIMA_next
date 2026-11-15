from __future__ import annotations

from io import StringIO

import pytest

from demo_cli.config import TraceMode
from demo_cli.diagnostics import adapt_response
from demo_cli.renderer import Renderer
from runtime import (
    ExecutionOutcome,
    ExecutionProfile,
    ExecutionStatus,
    PrimaResponse,
    RuntimeComponent,
    RuntimeDiagnostics,
    TaskKind,
)


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
