"""One bounded provider session shared by every campaign benchmark."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from benchmarks.campaign.config import ProviderConfig
from config.runtime_config import RuntimeConfig
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

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds

    def validate(self, request: LLMRequest) -> None:
        if request.response_schema and request.generation.structured_output is not StructuredOutputMode.JSON_SCHEMA:
            raise ValueError("response schema requires JSON_SCHEMA mode")

    def send(self, request: LLMRequest) -> LLMResponse:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if "GoEmotions taxonomy" in request.prompt:
            text = '{"labels":["joy"]}'
        elif request.response_schema:
            text = '{"answer":"Alpha","evidence":[],"insufficient_information":false}'
        else:
            text = "Alpha"
        prompt_tokens = len(request.prompt.split())
        return LLMResponse(
            text=text,
            raw={"fake": True},
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": len(text.split())},
            provider=self.name,
        )


class SharedProviderSession:
    """Own one client and enforce the campaign-wide active inference bound."""

    def __init__(
        self,
        config: ProviderConfig,
        max_active_requests: int,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        started = time.perf_counter()
        provider: Any = (
            FakeProvider(config.fake_delay_seconds)
            if config.kind == "fake"
            else ProviderFactory.get_provider(config.kind)
        )
        if config.endpoint and hasattr(provider, "base_url"):
            provider.base_url = config.endpoint
        self.provider = _BoundedProvider(provider, max_active_requests)
        self.client = LLMClient(provider_name=config.kind, provider=self.provider)
        self.session_initialization_ms = (time.perf_counter() - started) * 1000
        self.runtime_config = runtime_config or RuntimeConfig.from_environment()

    def runtime_factory(self, **kwargs: Any) -> PrimaRuntime:
        """Build isolated runtimes without constructing another model client."""

        kwargs["llm_client"] = self.client
        kwargs["runtime_config"] = self.runtime_config
        return PrimaRuntime(**kwargs)

    def telemetry(self) -> dict[str, Any]:
        return {
            "provider_session_initialization_ms": self.session_initialization_ms,
            "provider_session_initialization_measurement": "client/provider object construction; not model load",
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
        self._cache: dict[str, LLMResponse] = {}
        self._cache_hits = 0
        self._first_started: float | None = None
        self._last_finished: float | None = None

    def validate(self, request: LLMRequest) -> None:
        self.provider.validate(request)

    def send(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        cache_key = json.dumps(
            {
                "prompt": request.prompt,
                "system_prompt": request.system_prompt,
                "schema": request.response_schema,
                "generation": request.generation.to_dict(),
            },
            sort_keys=True,
            default=str,
        )
        with self._semaphore:
            with self._lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache_hits += 1
                    return cached
                if self._first_started is None:
                    self._first_started = started
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
                    self._cache[cache_key] = response
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
                    self._last_finished = time.perf_counter()

    def telemetry(self) -> dict[str, Any]:
        with self._lock:
            latencies = sorted(self._latencies)
            attempts, successes = self._attempts, self._successes
            prompt, completion = self._prompt_tokens, self._completion_tokens
            maximum = self._max_active
            retries, timeouts = self._retries, self._timeouts
            cache_hits = self._cache_hits
            wall_seconds = (
                self._last_finished - self._first_started
                if self._first_started is not None and self._last_finished is not None
                else 0.0
            )
        return {
            "request_attempts": attempts,
            "successful_requests": successes,
            "retries": retries,
            "timeouts": timeouts,
            "response_cache_hits": cache_hits,
            "max_active_requests": maximum,
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "p99": _percentile(latencies, 0.99),
            },
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "inference_wall_time_seconds": wall_seconds,
            "throughput_requests_per_second": successes / wall_seconds if wall_seconds else 0.0,
        }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    return values[max(0, min(len(values) - 1, int(round((len(values) - 1) * fraction))))]
