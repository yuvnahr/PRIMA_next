"""Lightweight types used across the LLM gateway."""
from dataclasses import dataclass
from typing import Any

from llm.generation_config import GenerationConfig


@dataclass
class LLMRequest:
    prompt: str
    generation: GenerationConfig
    metadata: dict[str, Any] | None = None
    system_prompt: str | None = None
    response_schema: dict[str, Any] | None = None

    @property
    def model(self) -> str:
        return self.generation.model


@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    usage: dict[str, int] | None = None
    provider: str | None = None
