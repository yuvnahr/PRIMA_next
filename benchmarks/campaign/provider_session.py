"""One bounded provider session shared by every campaign benchmark."""

from __future__ import annotations

import threading
import time
from typing import Any

from benchmarks.campaign.config import ProviderConfig
from llm.generation_config import ProviderCapabilities, StructuredOutputMode
from llm.llm_client import LLMClient
from llm.llm_types import LLMRequest, LLMResponse
from llm.provider import Provider, ProviderError, ProviderFactory
from runtime.prima_runtime import PrimaRuntime


class FakeProvider(Provider):
    """Deterministic no-network provider for the checked-in smoke campaign."""

    name = "fake"
    capabilities = ProviderCapabilities(
        system_prompts=True,
        structured_output=(StructuredOutputMode.NONE, StructuredOutputMode.JSON_OBJECT, StructuredOutputMode.JSON_SCHEMA),
        token_accounting=True,
        retries=True,
        seed=True,
        supported_fields=frozenset({"temperature", "top_p", "top_k", "repeat_penalty", "seed", "max_output_tokens"}),
    )

    def __init__(self) -> None:
        pass

    def validate(self, request: LLMRequest) -> None:
        if request.response_schema and request.generation.structured_output is not StructuredOutputMode.JSON_SCHEMA:
            raise ValueError("response schema requires JSON_SCHEMA mode")

    def send(self, request: LLMRequest) -> LLMResponse:
        text = '{"labels":["joy"]}' if "GoEmotions taxonomy" in request.prompt else "Alpha"
        prompt_tokens = len(request.prompt.split())
        return LLMResponse(
            text=text,
            raw={"fake": True},
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": len(text.split())},
            provider=self.name,
        )


class SharedProviderSession:
    """Own one client and enforce the campaign-wide active inference bound."""

    def __init__(self, config: ProviderConfig, max_active_requests: int) -> None:
        started = time.perf_counter()
        provider: Any = FakeProvider() if config.kind == "fake" else ProviderFactory.get_provider(config.kind)
        if config.endpoint and hasattr(provider, "base_url"):
            provider.base_url = config.endpoint
        self.provider = _BoundedProvider(provider, max_active_requests)
        self.client = LLMClient(provider_name=config.kind, provider=self.provider)
        self.model_load_time_ms = (time.perf_counter() - started) * 1000

    def runtime_factory(self, **kwargs: Any) -> PrimaRuntime:
        """Build isolated runtimes without constructing another model client."""

        kwargs["llm_client"] = self.client
        return PrimaRuntime(**kwargs)

    def telemetry(self) -> dict[str, Any]:
        return {
            "model_load_time_ms": self.model_load_time_ms,
            "model_load_measurement": "shared provider-session initialization",
            **self.provider.telemetry(),
        }


class _BoundedProvider(Provider):
    def __init__(self, provider: Provider, limit: int) -> None:
        self.provider = provider
        self.name = str(getattr(provider, "name", "unknown"))
        self.capabilities = provider.capabilities
        self._semaphore = threading.BoundedSemaphore(limit)
        self._lock = threading.Lock()
        self._active = self._max_active = self._attempts = self._successes = self._timeouts = 0
        self._retries = 0
        self._request_local = threading.local()
        self._latencies: list[float] = []
        self._prompt_tokens = self._completion_tokens = 0

    def validate(self, request: LLMRequest) -> None:
        self.provider.validate(request)

    def send(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        with self._semaphore:
            with self._lock:
                self._attempts += 1
                if getattr(self._request_local, "last", None) is request:
                    self._retries += 1
                self._request_local.last = request
                self._active += 1
                self._max_active = max(self._max_active, self._active)
            try:
                response: LLMResponse = self.provider.send(request)
                with self._lock:
                    self._successes += 1
                    usage = response.usage or {}
                    self._prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                    self._completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                return response
            except (ProviderError, TimeoutError) as exc:
                with self._lock:
                    if "timeout" in str(exc).casefold() or "timed out" in str(exc).casefold():
                        self._timeouts += 1
                raise
            finally:
                elapsed = (time.perf_counter() - started) * 1000
                with self._lock:
                    self._active -= 1
                    self._latencies.append(elapsed)

    def telemetry(self) -> dict[str, Any]:
        with self._lock:
            latencies = sorted(self._latencies)
            attempts, successes = self._attempts, self._successes
            prompt, completion = self._prompt_tokens, self._completion_tokens
            maximum = self._max_active
            retries, timeouts = self._retries, self._timeouts
        seconds = sum(latencies) / 1000
        return {
            "request_attempts": attempts,
            "successful_requests": successes,
            "retries": retries,
            "timeouts": timeouts,
            "max_active_requests": maximum,
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "p99": _percentile(latencies, 0.99),
            },
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "throughput_requests_per_second": successes / seconds if seconds else 0.0,
        }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    return values[max(0, min(len(values) - 1, int(round((len(values) - 1) * fraction))))]
