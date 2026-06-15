"""Model configuration helpers."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    name: str
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
