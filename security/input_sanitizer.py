"""Sanitize user-provided input to reduce prompt injection risk.

This is intentionally conservative and should be extended with policy-driven
rules for your application's threat model.
"""
import re
from re import Pattern

# Patterns that commonly indicate instruction-injection attempts. Keep this
# list conservative; it's not a replacement for runtime monitoring.
_INJECTION_PATTERNS: Pattern[str] = re.compile(
    r"(ignore (?:previous|prior) instructions|don't follow .*instructions|bypass|DROP TABLE|\bpassword\b|\bsecret\b)",
    flags=re.IGNORECASE,
)


def sanitize_input(value: str, max_length: int = 2000) -> str:
    """Return a sanitized copy of `value`.

    - strips control characters
    - removes common prompt-injection phrases
    - truncates to `max_length`
    """
    if value is None:
        return ""
    # remove non-printable/control characters
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
    # remove suspicious phrases
    cleaned = _INJECTION_PATTERNS.sub("[REDACTED]", cleaned)
    # normalize whitespace and truncate
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned
