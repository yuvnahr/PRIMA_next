"""Token counting utilities.

This provides a small, dependency-free fallback. For production use, integrate
with a provider-specific tokenizer (e.g. tiktoken) and keep this as a fallback.
"""
from typing import Optional


def count_tokens(text: Optional[str]) -> int:
    """Naive token counter (whitespace-based)."""
    if not text:
        return 0
    return len(text.split())
