from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from demo_cli import cli
from demo_cli.live_eval import ASPECTS, INPUT_LIMIT_BYTES, EvalUnavailable, judge_prompt, parse_evaluation
from llm.generation_config import StructuredOutputMode
from llm.llm_types import LLMResponse
from llm.provider import ProviderError
from tests.demo_cli.test_comparison import paired
from tests.demo_cli.test_session_cli import _settings


def scorecard():
    return {
        name: {aspect: {"score": value, "reason": f"{name} covered the request."} for aspect in ASPECTS}
        for name, value in (("A", 2), ("B", 4))
    }


def test_live_eval_uses_the_actual_follow_up_histories_and_caches_one_call(monkeypatch, capsys):
    session, provider = paired(monkeypatch)
    send = provider.send
    judge_requests = []

    def respond(request):
        if request.response_schema:
            judge_requests.append(request)
            return LLMResponse(json.dumps(scorecard()))
        return send(request)

    monkeypatch.setattr(provider, "send", respond)
    monkeypatch.setattr(cli, "get_settings", _settings)
    monkeypatch.setattr("demo_cli.comparison.ComparisonSession", lambda *args, **kwargs: session)
    commands = iter([
        "/eval", "Remember I have the offline PDF.", "/eval",
        "How can I use it if the network is down?", "/eval", "/eval",
        "/reset", "/eval", "Start a new plan.", "/eval", "/quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    assert cli.main(["compare", "--provider", "groq", "--no-color"]) == 0
    assert len(provider.requests) == 6 and len(judge_requests) == 3
    first, follow_up, fresh = (json.loads(request.prompt) for request in judge_requests)
    assert len(first) == 1 and len(follow_up) == 2 and len(fresh) == 1
    assert follow_up[0]["user"] == "Remember I have the offline PDF."
    assert follow_up[0]["A"] == first[0]["B"] and follow_up[0]["B"] == first[0]["A"]
    assert {follow_up[0]["A"], follow_up[0]["B"]} == {"BASE answer 1", "PRIMA answer 2"}
    assert follow_up[1]["user"] == "How can I use it if the network is down?"
    assert {follow_up[1]["A"], follow_up[1]["B"]} == {"BASE answer 3", "PRIMA answer 4"}
    assert all(not request.history and request.model == "openai/gpt-oss-20b" for request in judge_requests)
    assert all(request.generation.retries == 0 and request.generation.timeout_seconds == 15
               and request.generation.structured_output is StructuredOutputMode.JSON_SCHEMA for request in judge_requests)
    output = capsys.readouterr().out
    assert output.count("QUICK RUBRIC ESTIMATE") == 4
    assert output.count("Send a paired turn before using /eval.") == 2
    assert "Base 50/100 | PRIMA 100/100" in output and "Base 100/100 | PRIMA 50/100" in output


def test_rubric_uses_shared_applicability_and_raw_near_tie_threshold():
    cards = scorecard()
    cards["B"] = scorecard()["A"]
    for card in cards.values():
        card["context"] = None
    cards["B"]["task"]["score"] = 3
    result = parse_evaluation(json.dumps(cards), model="judge", base_first=False, secrets=())
    rows = dict(result.rows())
    assert rows["Aspects scored"] == "3/4"
    assert rows["Overall"] == "Base 58/100 | PRIMA 50/100"
    assert rows["Result"] == "Near tie"


@pytest.mark.parametrize("bad_grade", [True, 5, "4", None, {"score": 1, "reason": "x" * 141}])
def test_invalid_grade_rejects_the_pair(bad_grade):
    cards = scorecard()
    cards["B"]["task"] = bad_grade
    with pytest.raises(EvalUnavailable, match="invalid scorecard"):
        parse_evaluation(json.dumps(cards), model="judge", base_first=True, secrets=())


def test_context_na_must_match_and_reasons_are_redacted():
    cards = scorecard()
    cards["A"]["context"] = None
    with pytest.raises(EvalUnavailable, match="invalid scorecard"):
        parse_evaluation(json.dumps(cards), model="judge", base_first=True, secrets=())
    cards["B"]["context"] = None
    cards["B"]["task"]["reason"] = "Printed fixture-key."
    rows = parse_evaluation(json.dumps(cards), model="judge", base_first=True, secrets=("fixture-key",)).rows()
    assert ("PRIMA reason", "Printed [REDACTED].") in rows


def test_full_transcript_budget_counts_bytes_and_redacts_credentials():
    prompt = judge_prompt([("Use fixture-key", "No", "No")], base_first=True, secrets=("fixture-key",))
    assert json.loads(prompt)[0]["user"] == "Use [REDACTED]"
    with pytest.raises(EvalUnavailable, match="Full context"):
        judge_prompt([("é" * (INPUT_LIMIT_BYTES // 2), "A", "B")], base_first=True, secrets=())


def test_failed_pair_is_not_graded_and_judge_errors_do_not_replace_turn(monkeypatch):
    session, provider = paired(monkeypatch, flags=("--judge-model", "custom-judge"))

    async def run():
        try:
            turn = await session.send("Give me a practical tip.")
            session.latest_turn = replace(turn, prima=replace(turn.prima, status="failed"))
            with pytest.raises(EvalUnavailable, match="Both branches"):
                await session.evaluate_latest(lambda _text: pytest.fail("Judge should not run"))
            session.latest_turn = replace(turn, prima=replace(turn.prima, status="abstained",
                                                               text="I need more information to help."))
            calls = []

            def limited(request):
                calls.append(request)
                raise ProviderError("HTTP 429 fixture-key")

            monkeypatch.setattr(provider, "send", limited)
            with pytest.raises(EvalUnavailable, match="rate limit") as error:
                await session.evaluate_latest(lambda _text: None)
            assert "fixture-key" not in str(error.value)
            assert len(calls) == 1 and calls[0].model == "custom-judge"
            assert session.latest_turn.prima.status == "abstained"
            assert not session.needs_reset
        finally:
            await session.close()

    asyncio.run(run())
