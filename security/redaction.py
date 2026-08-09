"""Recursive diagnostics redaction for secrets, URLs and optional prompts."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_SECRET_MARKERS = ("api_key", "apikey", "authorization", "password", "secret", "access_token")
_PROMPT_MARKERS = ("prompt", "user_input", "raw_input")
_QUERY_SECRET_MARKERS = ("token", "key", "auth", "signature")


def redact(value: Any, *, redact_prompts: bool = True) -> Any:
    """Return a copy safe for diagnostics and logs."""

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in _SECRET_MARKERS):
                result[str(key)] = "[REDACTED]"
            elif redact_prompts and any(marker in lowered for marker in _PROMPT_MARKERS):
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = redact(item, redact_prompts=redact_prompts)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item, redact_prompts=redact_prompts) for item in value]
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        parts = urlsplit(value)
        query = urlencode(
            [
                (key, "[REDACTED]" if any(marker in key.lower() for marker in _QUERY_SECRET_MARKERS) else item)
                for key, item in parse_qsl(parts.query, keep_blank_values=True)
            ]
        )
        return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))
    return value
