"""Validate and optionally scrub model outputs before use or logging."""
import re
from typing import Pattern


# Patterns that likely indicate the model is leaking secrets or sensitive data.
_SENSITIVE_PATTERNS: Pattern = re.compile(
    r"(api[_-]?key|secret|password|private key|ssh-rsa|BEGIN RSA PRIVATE KEY|BEGIN PRIVATE KEY)",
    flags=re.IGNORECASE,
)


def validate_output(text: str) -> bool:
    """Raise a ValueError if the output contains likely secrets or disallowed content.

    Returns True when validation passes. The caller may catch and handle the
    ValueError (for example: redact, retry with a different prompt, or block).
    """
    if not text:
        return True
    if _SENSITIVE_PATTERNS.search(text):
        raise ValueError("LLM output contains potential secrets or sensitive data")
    return True
