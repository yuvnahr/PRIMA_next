"""Lightweight types used across the LLM gateway."""
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMRequest:
    model: str
    prompt: str
    temperature: float = 0.0
    max_tokens: int | None = None
    metadata: dict[str, Any] | None = None
    system_prompt: str | None = None
    response_format: dict[str, Any] | str | None = None


@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    usage: dict[str, int] | None = None
    provider: str | None = None
