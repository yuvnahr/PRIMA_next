"""Model configuration helpers."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    max_tokens: int | None = None
    context_window: int | None = None
