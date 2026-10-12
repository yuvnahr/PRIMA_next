from __future__ import annotations

import traceback
import urllib.error
from io import BytesIO
from types import SimpleNamespace

import pytest

from config.settings import Settings
from llm.generation_config import GenerationConfig, StructuredOutputMode
from llm.llm_client import LLMClient
from llm.llm_types import LLMRequest
from llm.provider import GroqProvider, ProviderCapabilityError, ProviderError, ProviderFactory


def _settings(**overrides):
    return SimpleNamespace(**{
        "groq_url": "https://api.groq.com/openai",
        "groq_api_key": "fixture-groq-key",
        "rate_limit_per_minute": 60,
        **overrides,
    })


@pytest.mark.parametrize("base_url", ["https://api.groq.com/openai", "https://groq.example/openai/"])
@pytest.mark.parametrize("mode", list(StructuredOutputMode))
def test_groq_selection_payload_and_response(monkeypatch, base_url, mode) -> None:
    captured = {}

    def fake_post(url, payload, headers, timeout):
        captured.update(url=url, payload=payload, headers=headers, timeout=timeout)
        return {
            "choices": [{"message": {"content": "generated"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3},
        }

    monkeypatch.setattr("llm.provider.post_json", fake_post)
    settings = _settings(groq_url=base_url)
    provider = ProviderFactory.get_provider("GROQ", settings=settings)
    assert isinstance(provider, GroqProvider)
    config = GenerationConfig(
        model="openai/gpt-oss-120b", provider="groq", temperature=0.3,
        top_p=0.8, seed=7, max_output_tokens=512, timeout_seconds=12, structured_output=mode,
    )
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    response = LLMClient(provider_name="groq", settings=settings, provider=provider).chat(
        "question", config, system_prompt="trusted policy",
        response_schema=schema if mode is StructuredOutputMode.JSON_SCHEMA else None,
    )
    assert captured["url"] == f"{base_url.rstrip('/')}/v1/chat/completions"
    assert captured["headers"] == {
        "Authorization": "Bearer fixture-groq-key", "Content-Type": "application/json",
    }
    assert captured["timeout"] == 12
    payload = captured["payload"]
    assert payload["model"] == "openai/gpt-oss-120b"
    assert payload["messages"] == [
        {"role": "system", "content": "trusted policy"}, {"role": "user", "content": "question"},
    ]
    assert (payload["temperature"], payload["max_tokens"], payload["top_p"], payload["seed"]) == (0.3, 512, 0.8, 7)
    if mode is StructuredOutputMode.NONE:
        assert "response_format" not in payload
    elif mode is StructuredOutputMode.JSON_OBJECT:
        assert payload["response_format"] == {"type": "json_object"}
    else:
        assert payload["response_format"]["json_schema"]["schema"] == schema
    assert response.provider == "groq"
    assert response.text == "generated"
    assert response.usage == {"prompt_tokens": 8, "completion_tokens": 3}


def test_groq_settings_follow_environment(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "fixture-env-key")
    monkeypatch.setenv("GROQ_URL", "https://groq.example/openai")
    monkeypatch.setenv("DEFAULT_PROVIDER", "groq")
    monkeypatch.setenv("DEFAULT_MODEL", "openai/gpt-oss-120b")
    settings = Settings(_env_file=None)
    assert settings.groq_api_key == "fixture-env-key"
    assert settings.groq_url == "https://groq.example/openai"
    assert settings.default_provider == "groq"
    assert settings.default_model == "openai/gpt-oss-120b"
    monkeypatch.delenv("GROQ_URL")
    assert Settings(_env_file=None).groq_url == "https://api.groq.com/openai"


def test_groq_missing_key_and_unsupported_fields_fail_before_http(monkeypatch) -> None:
    def unexpected_http(*args, **kwargs):
        pytest.fail("Invalid configuration must not send HTTP")

    monkeypatch.setattr("llm.provider.post_json", unexpected_http)
    config = GenerationConfig(model="openai/gpt-oss-120b", provider="groq")
    with pytest.raises(ProviderError, match="groq API key"):
        GroqProvider(_settings(groq_api_key=None)).send(LLMRequest("question", config))
    with pytest.raises(ProviderCapabilityError, match="top_k"):
        GroqProvider(_settings()).send(LLMRequest("question", config.with_overrides(top_k=3)))


@pytest.mark.parametrize("status", [401, 429, 500, None])
def test_groq_requests_errors_preserve_status_without_secrets(monkeypatch, status) -> None:
    import requests

    def fail_post(*args, **kwargs):
        response = requests.Response()
        response.status_code = status or 0
        raise requests.exceptions.HTTPError(
            "Authorization: Bearer fixture-groq-key", response=response if status else None,
        )

    monkeypatch.setattr("llm.provider.requests", requests)
    monkeypatch.setattr(requests, "post", fail_post)
    client = LLMClient(provider_name="groq", settings=_settings())
    with pytest.raises(ProviderError) as caught:
        client.chat("question", GenerationConfig(model="openai/gpt-oss-120b", provider="groq"))
    assert "fixture-groq-key" not in str(caught.value)
    assert "fixture-groq-key" not in "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert str(status) in str(caught.value) if status else "HTTPError" in str(caught.value)


@pytest.mark.parametrize("http_error", [True, False])
def test_groq_urllib_errors_do_not_expose_keys(monkeypatch, http_error) -> None:
    def fail_open(request, timeout):
        assert request.get_header("Authorization") == "Bearer fixture-groq-key"
        if http_error:
            raise urllib.error.HTTPError(request.full_url, 401, "fixture-groq-key", None, None)
        raise urllib.error.URLError("fixture-groq-key")

    monkeypatch.setattr("llm.provider.requests", None)
    monkeypatch.setattr("urllib.request.urlopen", fail_open)
    with pytest.raises(ProviderError) as caught:
        GroqProvider(_settings()).send(LLMRequest(
            "question", GenerationConfig(model="openai/gpt-oss-120b", provider="groq"),
        ))
    assert "fixture-groq-key" not in str(caught.value)
    assert "401" in str(caught.value) if http_error else "URLError" in str(caught.value)


@pytest.mark.parametrize("backend", ["requests", "urllib"])
def test_groq_invalid_json_uses_safe_provider_error(monkeypatch, backend) -> None:
    if backend == "requests":
        import requests

        response = requests.Response()
        response.status_code = 200
        response._content = b"fixture-groq-key"
        monkeypatch.setattr("llm.provider.requests", requests)
        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)
    else:
        monkeypatch.setattr("llm.provider.requests", None)
        monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: BytesIO(b"fixture-groq-key"))
    with pytest.raises(ProviderError) as caught:
        GroqProvider(_settings()).send(LLMRequest(
            "question", GenerationConfig(model="openai/gpt-oss-120b", provider="groq"),
        ))
    assert "fixture-groq-key" not in str(caught.value)
