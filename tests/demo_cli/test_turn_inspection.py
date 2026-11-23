from __future__ import annotations

import asyncio
from io import StringIO

from demo_cli.commands import dispatch_command
from demo_cli.renderer import Renderer
from demo_cli.turn_inspection import turn_rows
from llm.llm_types import LLMResponse
from tests.demo_cli.test_session_cli import _session


def test_inspections_read_the_recorded_turn_without_generation_or_state_changes():
    session, provider = _session(["--trace", "off"])
    commands = ("/affect", "/affect detail", "/decision", "/decision detail",
                "/evidence", "/memory why", "/action", "/review")

    async def inspect(command):
        output = StringIO()
        await dispatch_command(command, session, Renderer(output=output, no_color=True, width=120))
        return " ".join(output.getvalue().split())

    async def run():
        try:
            for command in commands:
                assert "No turn response is available yet." in await inspect(command)
            first = await session.send("Please remember that I prefer concise explanations.")
            affect = await inspect("/affect")
            for label, score in first.output_data["affect_state"]["emotions"].items():
                assert f"Score: {label} {score:.3f}" in affect
            assert "Heuristic confidence" in affect
            assert "State update committed" in affect
            assert "Salience" in await inspect("/affect detail")
            decision = await inspect("/decision")
            for entry in first.output_data["uncertainty"]["decision_probabilities"]:
                assert f"Estimator: {entry['decision_type']} {entry['probability']:.2%}" in decision
            for index, entry in enumerate(first.output_data["execution_decision_history"], 1):
                assert f"Gate {index} {entry['decision']} [{entry['rule']}]" in decision
            assert "Confidence interval" in await inspect("/decision detail")
            assert "Final context items 0" in await inspect("/evidence")
            assert "Stored yes" in await inspect("/memory why")
            assert "Stored memories" in await inspect("/memory")
            assert "Tool submissions 0" in await inspect("/action")
            assert "Technical verdict passed" in await inspect("/review")
            assert "Usage: /decision [detail]" in await inspect("/decision invalid")
            assert len(provider.requests) == 1
            assert session.runtime.state_manager.load(session.session_id).version == first.diagnostics.state_version

            await session.send("What type of explanations do I prefer?")
            evidence = await inspect("/evidence")
            assert "In context 1" in evidence and "concise explanations" in evidence
            session.set_profile("model")
            assert "Last turn profile prima_full" in await inspect("/affect")
            await session.send("Give me a practical tip.")
            for command in ("/affect", "/decision", "/evidence"):
                assert "skipped by profile" in await inspect(command)
            assert "Action phase not executed" in await inspect("/action")
            assert len(provider.requests) == 3
            await session.reset()
            assert "No turn response is available yet." in await inspect("/decision")
        finally:
            await session.close()

    asyncio.run(run())


def test_review_shows_a_real_failed_validation_and_accepted_correction(monkeypatch):
    session, provider = _session(["--profile", "model"])
    normal = provider.send
    calls = 0

    def once_invalid(request):
        nonlocal calls
        calls += 1
        return LLMResponse("") if calls == 1 else normal(request)

    monkeypatch.setattr(provider, "send", once_invalid)

    async def run():
        try:
            response = await session.send("Give me a practical tip.")
            review = dict(turn_rows(response, session.session_id, "/review"))
            assert review["Reflection phase"] == "skipped by profile"
            assert review["Correction 1"] == "post_execution / regenerate / accepted"
            assert review["Advice reason"] == "output_schema_failure"
            assert review["Technical verdict"] == "passed"
            assert review["Validation passes"] == "2"
            assert calls == 2
        finally:
            await session.close()

    asyncio.run(run())


def test_inspection_does_not_print_credentials_or_raw_tool_payloads():
    session, _ = _session()

    async def run():
        try:
            response = await session.send("I am worried about the presentation.")
            step = response.output_data["inspection"]["action"]["steps"][0]
            step.update(message="fixture-key", tool_result="private tool payload")
            rendered = str(turn_rows(response, session.session_id, "/action", secrets=session.observability.secrets))
            assert "[REDACTED]" in rendered
            assert "fixture-key" not in rendered
            assert "private tool payload" not in rendered
        finally:
            await session.close()

    asyncio.run(run())
