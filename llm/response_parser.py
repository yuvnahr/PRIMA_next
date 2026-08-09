"""Parsers that convert provider-specific responses into LLMResponse."""

import json
import re
from typing import Any

from llm.llm_types import LLMResponse

_THINKING_BLOCK = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)


def extract_answer(text: str) -> str:
    """Return the final answer from a local model response."""

    answer = _THINKING_BLOCK.sub("", str(text)).strip()
    if answer.startswith("```") and answer.endswith("```"):
        answer = answer.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(answer)
    except (TypeError, ValueError):
        return answer
    if isinstance(payload, dict) and "answer" in payload:
        value = payload["answer"]
        if value is None or str(value).strip().lower() in {"null", "none"}:
            return ""
        return str(value).strip()
    return answer


def parse_openai_response(raw: Any) -> LLMResponse:
    """Parse a typical OpenAI chat-completions REST response."""

    text = ""
    usage = None
    try:
        if isinstance(raw, dict):
            choices = raw.get("choices", [])
            if choices:
                first = choices[0]
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
        if isinstance(raw, dict) and isinstance(raw.get("content"), list):
            text = "".join(
                str(item.get("text", ""))
                for item in raw["content"]
                if isinstance(item, dict) and item.get("type") == "text"
            )
        else:
            text = (raw.get("text") or raw.get("output") or raw.get("response") or str(raw)) if isinstance(raw, dict) else str(raw)
    except Exception:
        text = str(raw)
    usage = raw.get("usage") if isinstance(raw, dict) and isinstance(raw.get("usage"), dict) else None
    return LLMResponse(text=text, raw=raw, usage=usage, provider=provider)
