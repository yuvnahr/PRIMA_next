"""Typed, immutable generation and provider capability contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class StructuredOutputMode(Enum):
    """Requested provider-level response constraint."""

    NONE = "none"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class FallbackPolicy(Enum):
    """Explicit behavior when generation cannot produce a valid answer."""

    ABSTAIN = "abstain"
    EXTRACTIVE = "extractive"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Request-scoped generation settings with no execution-time environment reads."""

    model: str
    provider: str
    revision: str | None = None
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    seed: int | None = None
    max_output_tokens: int = 128
    timeout_seconds: float = 60.0
    retries: int = 0
    structured_output: StructuredOutputMode = StructuredOutputMode.NONE
    fallback_policy: FallbackPolicy = FallbackPolicy.FAIL

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.provider.strip():
            raise ValueError("Generation model and provider must be configured.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.repeat_penalty is not None and self.repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be positive")
        if self.max_output_tokens < 1 or self.timeout_seconds <= 0 or self.retries < 0:
            raise ValueError("Token limit and timeout must be positive; retries cannot be negative.")

    def with_overrides(self, **values: Any) -> GenerationConfig:
        """Return a validated copy with explicit request overrides."""

        return replace(self, **{key: value for key, value in values.items() if value is not None})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic representation."""

        return {
            "model": self.model,
            "provider": self.provider,
            "revision": self.revision,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "seed": self.seed,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
            "structured_output": self.structured_output.value,
            "fallback_policy": self.fallback_policy.value,
        }


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Features a provider adapter guarantees rather than silently discards."""

    system_prompts: bool
    structured_output: tuple[StructuredOutputMode, ...]
    token_accounting: bool
    retries: bool
    seed: bool
    supported_fields: frozenset[str]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible capability diagnostics."""

        return {
            "system_prompts": self.system_prompts,
            "structured_output": [mode.value for mode in self.structured_output],
            "token_accounting": self.token_accounting,
            "retries": self.retries,
            "seed": self.seed,
            "supported_fields": sorted(self.supported_fields),
        }
