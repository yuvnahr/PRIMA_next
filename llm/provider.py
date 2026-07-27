"""Provider adapters and factory for the unified LLM gateway.

Important: this module intentionally avoids importing vendor SDKs like
`openai`, `anthropic`, or `ollama` so that the rest of the codebase never
depends on those packages directly. HTTP calls are used so adapters remain
lightweight and explicit about where secrets are read from.
"""
from __future__ import annotations

import logging
import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from llm.llm_types import LLMRequest, LLMResponse
from llm.response_parser import parse_generic_response, parse_openai_response

# Safe defaults for optional external modules
requests: Any = None
try:
    import requests as _requests_module
    requests = _requests_module
except Exception:
    requests = None


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: int = 60) -> Any:
    """POST JSON using requests when available, otherwise urllib."""

    if requests is not None:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        try:
            response.raise_for_status()
        except Exception as exc:
            raise ProviderError(f"HTTP request failed: {exc} - {response.text}") from exc
        return response.json()

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers or {"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"HTTP request failed: {exc} - {body}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"HTTP request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ProviderError(f"HTTP request timed out after {timeout} seconds") from exc


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

# default then attempt to load real settings provider
def get_settings() -> Any:
    return None
try:
    from config import settings as _settings
    get_settings = _settings.get_settings
except Exception as exc:  # pragma: no cover - config may be added separately
    logging.getLogger(__name__).debug(
        "config.settings unavailable; using default settings stub: %s", exc
    )


class ProviderError(RuntimeError):
    pass


class Provider(ABC):
    """Abstract provider interface."""

    name: str = "generic"

    def __init__(self, settings: Any | None = None) -> None:
        self.settings = settings or get_settings()

    @abstractmethod
    def send(self, request: LLMRequest) -> LLMResponse:
        ...


class OpenAICompatibleProvider(Provider):
    """Provider for local servers exposing `/v1/chat/completions`."""

    base_url_env: str = ""
    default_base_url: str = ""
    default_model: str = ""

    def send(self, request: LLMRequest) -> LLMResponse:
        base = os.getenv(self.base_url_env) or self.default_base_url
        if not base:
            raise ProviderError(f"{self.name} base URL is not configured")
        model = request.model or os.getenv(f"{self.name.upper()}_MODEL") or self.default_model
        if not model:
            raise ProviderError(f"{self.name} model is not configured")
        url = f"{base.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = int(request.max_tokens)

        try:
            return parse_openai_response(post_json(url, payload, timeout=60))
        except ProviderError as exc:
            raise ProviderError(f"{self.name} request failed: {exc}") from exc


class OpenAIProvider(Provider):
    name = "openai"

    def send(self, request: LLMRequest) -> LLMResponse:
        key = None
        if self.settings is not None:
            key = getattr(self.settings, "openai_api_key", None)
        key = key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ProviderError("OpenAI API key not configured in environment or settings")

        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = int(request.max_tokens)

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            return parse_openai_response(post_json(url, payload, headers=headers, timeout=30))
        except ProviderError as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc


class AnthropicProvider(Provider):
    name = "anthropic"

    def send(self, request: LLMRequest) -> LLMResponse:
        key = None
        if self.settings is not None:
            key = getattr(self.settings, "anthropic_api_key", None)
        key = key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("Anthropic API key not configured in environment or settings")

        # Anthropic provides a REST endpoint; this is a minimal wrapper.
        url = "https://api.anthropic.com/v1/complete"
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        headers = {"x-api-key": key, "Content-Type": "application/json"}
        try:
            return parse_generic_response(post_json(url, payload, headers=headers, timeout=30), provider="anthropic")
        except ProviderError as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc


class OllamaProvider(OpenAICompatibleProvider):
    name = "ollama"
    base_url_env = "OLLAMA_URL"
    default_base_url = "http://localhost:11434"
    default_model = "llama3.1"

    def send(self, request: LLMRequest) -> LLMResponse:
        if os.getenv("OLLAMA_OPENAI_API", "").lower() in {"1", "true", "yes"}:
            return super().send(request)

        base = getattr(self.settings, "ollama_url", None) if self.settings is not None else None
        base = base or os.getenv("OLLAMA_URL") or self.default_base_url
        url = f"{base.rstrip('/')}/api/generate"
        payload = {
            "model": request.model or self.default_model,
            "prompt": request.prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": env_float("PRIMA_ANSWER_TEMPERATURE", request.temperature),
                "top_p": env_float("PRIMA_ANSWER_TOP_P", 0.8),
                "top_k": env_int("PRIMA_ANSWER_TOP_K", 40),
                "repeat_penalty": env_float("PRIMA_ANSWER_REPEAT_PENALTY", 1.1),
                "num_predict": env_int("PRIMA_ANSWER_MAX_TOKENS", int(request.max_tokens or 64)),
            },
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt
        if request.response_format:
            payload["format"] = request.response_format
        try:
            return parse_generic_response(post_json(url, payload, timeout=180), provider="ollama")
        except ProviderError as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc


class LMStudioProvider(OpenAICompatibleProvider):
    name = "lmstudio"
    base_url_env = "LM_STUDIO_URL"
    default_base_url = "http://localhost:1234"
    default_model = "local-model"


class LocalProvider(OllamaProvider):
    """Alias for local/hosted models. Uses the Ollama-compatible local API by default."""


class ProviderFactory:
    _map: dict[str, type[Provider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "lmstudio": LMStudioProvider,
        "lm-studio": LMStudioProvider,
        "local": LocalProvider,
    }

    @classmethod
    def get_provider(cls, name: str, settings: Any | None = None) -> Provider:
        name = (name or "").lower()
        provider_cls = cls._map.get(name)
        if not provider_cls:
            raise ProviderError(f"Unknown provider: {name}")
        return provider_cls(settings=settings)
