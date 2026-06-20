"""Token counting utilities.

This provides a small, dependency-free fallback. For production use, integrate
with a provider-specific tokenizer (e.g. tiktoken) and keep this as a fallback.
"""


def count_tokens(text: str | None) -> int:
    """Naive token counter (whitespace-based)."""
    if not text:
        return 0
    return len(text.split())
