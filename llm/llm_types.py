"""Lightweight types used across the LLM gateway."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMRequest:
    model: str
    prompt: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    usage: Optional[Dict[str, int]] = None
    provider: Optional[str] = None
