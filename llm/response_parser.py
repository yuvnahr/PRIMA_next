"""Parsers that convert provider-specific responses into LLMResponse."""
from typing import Any

from llm.llm_types import LLMResponse


def parse_openai_response(raw: Any) -> LLMResponse:
    """Parse a typical OpenAI chat-completions REST response.

    This function is defensive — it accepts raw dicts or already-stringified
    responses and returns a normalized LLMResponse.
    """
    text = ""
    usage = None
    try:
        if isinstance(raw, dict):
            choices = raw.get("choices", [])
            if choices:
                # Chat completion structure
                first = choices[0]
                # prefer message.content, fallback to text
                text = (first.get("message", {}) or {}).get("content") or first.get("text") or ""
            usage = raw.get("usage")
        else:
            text = str(raw)
    except Exception:
        text = str(raw)
    return LLMResponse(text=text or "", raw=raw, usage=usage, provider="openai")


def parse_generic_response(raw: Any, provider: str = "generic") -> LLMResponse:
    """Fallback parser for other providers."""
    try:
        if isinstance(raw, dict):
            # attempt common fields
            text = raw.get("text") or raw.get("output") or raw.get("response") or str(raw)
        else:
            text = str(raw)
    except Exception:
        text = str(raw)
    return LLMResponse(text=text, raw=raw, provider=provider)
