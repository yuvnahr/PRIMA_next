"""Bounded, explicit content export; never stringify arbitrary runtime objects."""

from __future__ import annotations

import math
import re
from enum import Enum
from itertools import islice
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TEXT_LIMIT = 65_536
CONTENT_BUDGET = 131_072
ITEM_LIMIT = 64
DEPTH_LIMIT = 10
_SECRET_KEY = re.compile(r"api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token", re.I)
_CREDENTIAL = re.compile(
    r"\b(?:Bearer|Basic)\s+[^\s,;\"<>]+|\b(?:gsk_|sk-)[A-Za-z0-9_-]{12,}|"
    r"\b(?:api[_-]?key|authorization|password|secret|access[_-]?token)[\"']?\s*[=:]\s*"
    r"(?:(?:Bearer|Basic)\s+)?(?:\"[^\"]*\"|'[^']*'|[^\s,;\"'<>]+)",
    re.I,
)
_URL = re.compile(r"https?://[^\s<>\"']+", re.I)


def sanitize_content(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    """Return JSON values with visible limits and no configured credentials."""
    remaining = CONTENT_BUDGET
    seen: set[int] = set()

    def clean_url(match: re.Match[str]) -> str:
        try:
            parts = urlsplit(match.group())
            host = parts.netloc.rsplit("@", 1)[-1]
            query = urlencode([(key, "[REDACTED]" if re.search(r"token|key|auth|signature", key, re.I) else item)
                               for key, item in parse_qsl(parts.query, keep_blank_values=True)])
            return urlunsplit((parts.scheme, host, parts.path, query, ""))
        except ValueError:
            return "[REDACTED URL]"

    def clean_text(text: str) -> str:
        for secret in sorted(secrets, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
        return _CREDENTIAL.sub("[REDACTED]", _URL.sub(clean_url, text))

    def visit(item: Any, depth: int = 0) -> Any:
        nonlocal remaining
        if remaining <= 0 or depth > DEPTH_LIMIT:
            return {"truncated": True, "reason": "content_budget" if remaining <= 0 else "depth_limit"}
        remaining -= 16
        if isinstance(item, Enum):
            return visit(item.value, depth)
        if item is None or isinstance(item, bool | int):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, str):
            text = clean_text(item)
            limit = max(0, min(TEXT_LIMIT, remaining))
            remaining -= min(len(text), limit)
            if len(text) > limit:
                return {"text": text[:limit], "truncated": True, "original_characters": len(item)}
            return text
        if isinstance(item, dict | list | tuple):
            if id(item) in seen:
                return {"unavailable": "cycle"}
            seen.add(id(item))
            try:
                if isinstance(item, dict):
                    result = {}
                    for key, entry in islice(item.items(), ITEM_LIMIT):
                        if not isinstance(key, str):
                            continue
                        safe_key = clean_text(key)[:200]
                        remaining -= len(safe_key)
                        result[safe_key] = "[REDACTED]" if _SECRET_KEY.search(key) else visit(entry, depth + 1)
                        if remaining <= 0:
                            break
                    if len(item) > len(result):
                        result["_capture_limit"] = {"truncated": True, "original_items": len(item)}
                    return result
                entries = []
                for entry in islice(item, ITEM_LIMIT):
                    entries.append(visit(entry, depth + 1))
                    if remaining <= 0:
                        break
                if len(item) > len(entries):
                    entries.append({"truncated": True, "original_items": len(item)})
                return entries
            finally:
                seen.remove(id(item))
        return {"unavailable": type(item).__name__}

    return visit(value)
