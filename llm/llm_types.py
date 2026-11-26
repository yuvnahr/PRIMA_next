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
    history: tuple[dict[str, str], ...] = ()

    def __post_init__(self) -> None:
        if len(self.history) % 2:
            raise ValueError("Chat history must contain complete user/assistant pairs.")
        for index, message in enumerate(self.history):
            expected = "user" if index % 2 == 0 else "assistant"
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] != expected or not isinstance(message["content"], str)
                    or not message["content"].strip()):
                raise ValueError("Chat history must alternate nonempty user and assistant messages.")
        self.history = tuple(dict(message) for message in self.history)

    def chat_messages(self) -> list[dict[str, str]]:
        """The logical messages shared by transport and content tracing."""
        messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        return messages + [dict(message) for message in self.history] + [{"role": "user", "content": self.prompt}]

    @property
    def model(self) -> str:
        return self.generation.model


@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    usage: dict[str, int] | None = None
    provider: str | None = None
