"""Provider adapters and factory for the unified LLM gateway.

Important: this module intentionally avoids importing vendor SDKs like
`openai`, `anthropic`, or `ollama` so that the rest of the codebase never
depends on those packages directly. HTTP calls are used so adapters remain
lightweight and explicit about where secrets are read from.
"""
from __future__ import annotations

import os
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

# default then attempt to load real settings provider
def get_settings() -> Any:
    return None
try:
    from config import settings as _settings
    get_settings = _settings.get_settings
except Exception:  # pragma: no cover - config may be added separately
    pass


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

        if requests is None:
            raise ProviderError("requests library is required for HTTP providers but is not installed")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise ProviderError(f"OpenAI request failed: {e} - {resp.text}")
        raw = resp.json()
        return parse_openai_response(raw)


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
        if requests is None:
            raise ProviderError("requests library is required for HTTP providers but is not installed")
        headers = {"x-api-key": key, "Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise ProviderError(f"Anthropic request failed: {e} - {resp.text}")
        raw = resp.json()
        return parse_generic_response(raw, provider="anthropic")


class OllamaProvider(Provider):
    name = "ollama"

    def send(self, request: LLMRequest) -> LLMResponse:
        # Ollama commonly runs locally; read URL from settings or env
        base = None
        if self.settings is not None:
            base = getattr(self.settings, "ollama_url", None)
        base = base or os.getenv("OLLAMA_URL") or "http://localhost:11434"

        url = f"{base.rstrip('/')}/api/generate"
        payload = {"model": request.model, "prompt": request.prompt}
        if requests is None:
            raise ProviderError("requests library is required for HTTP providers but is not installed")
        resp = requests.post(url, json=payload, timeout=30)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise ProviderError(f"Ollama request failed: {e} - {resp.text}")
        raw = resp.json()
        return parse_generic_response(raw, provider="ollama")


class LocalProvider(OllamaProvider):
    """Alias for local/hosted models. Uses the Ollama-compatible local API by default."""


class ProviderFactory:
    _map: dict[str, type[Provider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "local": LocalProvider,
    }

    @classmethod
    def get_provider(cls, name: str, settings: Any | None = None) -> Provider:
        name = (name or "").lower()
        provider_cls = cls._map.get(name)
        if not provider_cls:
            raise ProviderError(f"Unknown provider: {name}")
        return provider_cls(settings=settings)
