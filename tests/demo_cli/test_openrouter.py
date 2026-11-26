import json

from config.settings import Settings
from demo_cli import cli
from demo_cli.live_eval import ASPECTS
from monitoring.tracing import Tracing


def test_openrouter_compare_and_eval_use_the_same_endpoint(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-openrouter-key")
    monkeypatch.setenv("OPENROUTER_UPSTREAM", "darkbloom/fp8")
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    settings = Settings(_env_file=None, langfuse_enabled=False)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    calls = []

    def post(url, payload, *, headers, timeout):
        calls.append((url, payload, headers))
        card = {aspect: {"score": 3, "reason": "Provides a usable answer."} for aspect in ASPECTS}
        answer = json.dumps({"A": card, "B": card}) if "response_format" in payload else "Open the offline PDF."
        return {"choices": [{"message": {"content": answer}}], "usage": {"total_tokens": 10}}

    monkeypatch.setattr("llm.provider.post_json", post)
    inputs = iter(["Give me one practical tip.", "Can you repeat it?", "/eval", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert cli.main(["compare", "--provider", "openrouter", "--temperature", "0.3", "--no-color"]) == 0
    assert len(calls) == 5
    for url, payload, headers in calls:
        assert url == "https://openrouter.ai/api/v1/chat/completions"
        assert headers["Authorization"] == "Bearer fixture-openrouter-key"
        assert payload["model"] == "openai/gpt-oss-20b"
        assert payload["provider"] == {"require_parameters": True, "only": ["darkbloom/fp8"], "allow_fallbacks": False}
    assert all(payload["temperature"] == 0.3 for _, payload, _ in calls[:4])
    assert calls[2][1]["messages"][2] == {"role": "assistant", "content": "Open the offline PDF."}
    assert calls[4][1]["response_format"]["type"] == "json_schema"
    output = capsys.readouterr().out
    assert "QUICK RUBRIC ESTIMATE" in output and "Base 75/100 | PRIMA 75/100" in output
    assert "fixture-openrouter-key" not in output
    assert "fixture-openrouter-key" in Tracing(settings=settings).secrets


def test_openrouter_preflight_reports_missing_key_without_network(monkeypatch, capsys):
    settings = Settings(_env_file=None, openrouter_api_key=None, langfuse_enabled=False)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    assert cli.main(["preflight", "--provider", "openrouter", "--no-color"]) == 1
    assert "Set OPENROUTER_API_KEY" in capsys.readouterr().out
