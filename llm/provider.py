"""Explicit HTTP provider adapters for the reusable LLM gateway."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlsplit

from config.settings import Settings, get_settings
from llm.generation_config import ProviderCapabilities, StructuredOutputMode
from llm.llm_types import LLMRequest, LLMResponse
from llm.response_parser import parse_generic_response, parse_openai_response

try:
    import requests
except ImportError:  # pragma: no cover - urllib is the supported dependency-free path
    requests = None  # type: ignore[assignment]


class ProviderError(RuntimeError):
    """Provider transport or configuration failure."""


class ProviderCapabilityError(ProviderError):
    """A request asks for a feature the selected provider cannot guarantee."""


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> Any:
    """POST JSON using requests when available, otherwise urllib."""

    if urlsplit(url).scheme not in {"http", "https"}:
        raise ProviderError("Provider URL must use http or https")
    if requests is not None:
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            raise ProviderError(f"HTTP request failed: {exc}") from exc
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310
        url,
        data=data,
        headers=headers or {"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310  # nosec B310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"HTTP request failed: {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ProviderError(f"HTTP request failed: {exc}") from exc


class Provider(ABC):
    """Provider session configured once when the client is built."""

    name = "generic"
    capabilities: ProviderCapabilities

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def validate(self, request: LLMRequest) -> None:
        """Reject unsupported fields instead of silently dropping them."""

        config = request.generation
        if request.system_prompt and not self.capabilities.system_prompts:
            raise ProviderCapabilityError(f"{self.name} does not support system prompts")
        if config.structured_output not in self.capabilities.structured_output:
            raise ProviderCapabilityError(
                f"{self.name} does not support structured output mode "
                f"'{config.structured_output.value}'"
            )
        if request.response_schema and config.structured_output is not StructuredOutputMode.JSON_SCHEMA:
            raise ProviderCapabilityError("A response schema requires structured_output='json_schema'.")
        optional = {
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repeat_penalty": config.repeat_penalty,
            "seed": config.seed,
        }
        unsupported = [
            name for name, value in optional.items()
            if value is not None and name not in self.capabilities.supported_fields
        ]
        if unsupported:
            raise ProviderCapabilityError(
                f"{self.name} does not support generation fields: {', '.join(unsupported)}"
            )

    @abstractmethod
    def send(self, request: LLMRequest) -> LLMResponse:
        """Send one already-validated request."""


_OPENAI_CAPABILITIES = ProviderCapabilities(
    system_prompts=True,
    structured_output=(
        StructuredOutputMode.NONE,
        StructuredOutputMode.JSON_OBJECT,
        StructuredOutputMode.JSON_SCHEMA,
    ),
    token_accounting=True,
    retries=True,
    seed=True,
    supported_fields=frozenset({"temperature", "top_p", "seed", "max_output_tokens"}),
)


class OpenAICompatibleProvider(Provider):
    """Provider for local servers exposing ``/v1/chat/completions``."""

    capabilities = _OPENAI_CAPABILITIES
    default_base_url = ""

    def __init__(self, settings: Settings | None = None, *, base_url: str | None = None) -> None:
        super().__init__(settings)
        self.base_url = base_url or self.default_base_url

    def send(self, request: LLMRequest) -> LLMResponse:
        self.validate(request)
        if not self.base_url:
            raise ProviderError(f"{self.name} base URL is not configured")
        config = request.generation
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_output_tokens,
        }
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.seed is not None:
            payload["seed"] = config.seed
        if config.structured_output is StructuredOutputMode.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
        elif config.structured_output is StructuredOutputMode.JSON_SCHEMA:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "prima_response", "schema": request.response_schema},
            }
        raw = post_json(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            payload,
            timeout=config.timeout_seconds,
        )
        response = parse_openai_response(raw)
        response.provider = self.name
        return response


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI chat-completions adapter."""

    name = "openai"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings, base_url=(settings or get_settings()).openai_url)
        self.api_key = self.settings.openai_api_key

    def send(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise ProviderError("OpenAI API key is not configured.")
        self.validate(request)
        config = request.generation
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_output_tokens,
        }
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.seed is not None:
            payload["seed"] = config.seed
        if config.structured_output is StructuredOutputMode.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
        elif config.structured_output is StructuredOutputMode.JSON_SCHEMA:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "prima_response", "schema": request.response_schema},
            }
        raw = post_json(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=config.timeout_seconds,
        )
        return parse_openai_response(raw)


class AnthropicProvider(Provider):
    """Anthropic Messages API adapter with explicit capability limits."""

    name = "anthropic"
    capabilities = ProviderCapabilities(
        system_prompts=True,
        structured_output=(StructuredOutputMode.NONE,),
        token_accounting=True,
        retries=True,
        seed=False,
        supported_fields=frozenset({"temperature", "top_p", "top_k", "max_output_tokens"}),
    )

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.base_url = self.settings.anthropic_url
        self.api_key = self.settings.anthropic_api_key

    def send(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise ProviderError("Anthropic API key is not configured.")
        self.validate(request)
        config = request.generation
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": config.max_output_tokens,
            "temperature": config.temperature,
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.top_k is not None:
            payload["top_k"] = config.top_k
        raw = post_json(
            f"{self.base_url.rstrip('/')}/v1/messages",
            payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=config.timeout_seconds,
        )
        return parse_generic_response(raw, provider=self.name)


class OllamaProvider(Provider):
    """Native Ollama generation adapter."""

    name = "ollama"
    capabilities = ProviderCapabilities(
        system_prompts=True,
        structured_output=(
            StructuredOutputMode.NONE,
            StructuredOutputMode.JSON_OBJECT,
            StructuredOutputMode.JSON_SCHEMA,
        ),
        token_accounting=True,
        retries=True,
        seed=True,
        supported_fields=frozenset(
            {"temperature", "top_p", "top_k", "repeat_penalty", "seed", "max_output_tokens"}
        ),
    )

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.base_url = self.settings.ollama_url

    def send(self, request: LLMRequest) -> LLMResponse:
        self.validate(request)
        config = request.generation
        options: dict[str, Any] = {
            "temperature": config.temperature,
            "num_predict": config.max_output_tokens,
        }
        for key in ("top_p", "top_k", "repeat_penalty", "seed"):
            value = getattr(config, key)
            if value is not None:
                options[key] = value
        payload: dict[str, Any] = {
            "model": config.model,
            "prompt": request.prompt,
            "stream": False,
            "think": False,
            "options": options,
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt
        if config.structured_output is StructuredOutputMode.JSON_OBJECT:
            payload["format"] = "json"
        elif config.structured_output is StructuredOutputMode.JSON_SCHEMA:
            payload["format"] = request.response_schema
        raw = post_json(
            f"{self.base_url.rstrip('/')}/api/generate",
            payload,
            timeout=config.timeout_seconds,
        )
        response = parse_generic_response(raw, provider=self.name)
        if isinstance(raw, dict):
            response.usage = {
                "prompt_tokens": int(raw.get("prompt_eval_count", 0)),
                "completion_tokens": int(raw.get("eval_count", 0)),
            }
        return response


class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio OpenAI-compatible adapter."""

    name = "lmstudio"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings, base_url=(settings or get_settings()).lmstudio_url)


class LocalProvider(OllamaProvider):
    """Alias for a configured local Ollama session."""


class ProviderFactory:
    """Construct a provider session once for an LLM client."""

    _map: dict[str, type[Provider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "lmstudio": LMStudioProvider,
        "lm-studio": LMStudioProvider,
        "local": LocalProvider,
    }

    @classmethod
    def get_provider(cls, name: str, settings: Settings | None = None) -> Provider:
        provider_cls = cls._map.get((name or "").lower())
        if provider_cls is None:
            raise ProviderError(f"Unknown provider: {name}")
        return provider_cls(settings=settings)
