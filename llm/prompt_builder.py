"""Prompt construction utilities.

This module ensures inputs are sanitized before being composed into prompts.
"""
from collections.abc import Mapping
from typing import Any

from .llm_types import LLMRequest

try:
    # import local sanitizer (kept internal to repo)
    from security.input_sanitizer import sanitize_input
except Exception:  # pragma: no cover - sanitizer should exist in the repo
    def sanitize_input(value: str, max_length: int = 2000) -> str:  # fallback no-op
        return value


def build_prompt(template: str, variables: Mapping[str, Any], model: str | None = None) -> LLMRequest:
    """Builds a sanitized prompt from a simple Python format template.

    Note: templates should be simple and not include untrusted format specifiers.
    """
    safe_vars = {k: sanitize_input(str(v)) for k, v in variables.items()}
    prompt = template.format(**safe_vars)
    return LLMRequest(model=model or safe_vars.get("model", ""), prompt=prompt)
