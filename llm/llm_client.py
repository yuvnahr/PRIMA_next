"""Reusable LLM client with explicit generation configuration."""

from __future__ import annotations

import time
from typing import Any

from config.settings import Settings, get_settings
from llm.generation_config import GenerationConfig, StructuredOutputMode
from llm.llm_types import LLMRequest, LLMResponse
from llm.provider import Provider, ProviderError, ProviderFactory
from llm.rate_limiter import RateLimiter
from security.output_validator import validate_output


class LLMClient:
    """Own one provider session and rate limiter for the runtime lifecycle."""

    def __init__(
        self,
        provider_name: str | None = None,
        settings: Settings | None = None,
        rate_limiter: RateLimiter | None = None,
        provider: Provider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider_name = provider_name or self.settings.default_provider
        self.provider = provider or ProviderFactory.get_provider(self.provider_name, settings=self.settings)
        self.rate_limiter = rate_limiter or RateLimiter(self.settings.rate_limit_per_minute)

    @property
    def capabilities(self) -> dict[str, Any]:
        """Expose the guarantees of the active provider session."""

        return self.provider.capabilities.to_dict()

    def chat(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send one request, retrying only according to the immutable config.

        The keyword compatibility parameters are retained for thin legacy callers;
        canonical runtime code passes ``generation_config``.
        """

        config = generation_config or GenerationConfig(
            model=str(model or self.settings.default_model),
            provider=self.provider_name,
            temperature=temperature,
            max_output_tokens=int(max_tokens or 128),
            structured_output=(
                StructuredOutputMode.JSON_SCHEMA
                if response_schema is not None or isinstance(response_format, dict)
                else StructuredOutputMode.JSON_OBJECT
                if response_format == "json"
                else StructuredOutputMode.NONE
            ),
        )
        if config.provider.lower() != self.provider_name.lower():
            raise ProviderError(
                f"Injected client is configured for '{self.provider_name}', not '{config.provider}'."
            )
        request = LLMRequest(
            prompt=prompt,
            generation=config,
            system_prompt=system_prompt,
            response_schema=response_schema or (response_format if isinstance(response_format, dict) else None),
        )
        self.provider.validate(request)
        last_error: ProviderError | None = None
        for attempt in range(config.retries + 1):
            if not self.rate_limiter.allow():
                raise RuntimeError("Rate limit exceeded")
            try:
                response = self.provider.send(request)
                validate_output(response.text)
                return response
            except ProviderError as exc:
                last_error = exc
                if attempt < config.retries:
                    time.sleep(min(0.05 * (2**attempt), 0.5))
        raise ProviderError(
            f"{self.provider_name} request failed after {config.retries + 1} attempt(s): {last_error}"
        ) from last_error
