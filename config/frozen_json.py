"""Integrity-checked immutable JSON artifacts with optional independent pins."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def freeze_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Refuse overwriting an experiment identity, including an internally invalid one."""
    sealed = {**value, "fingerprint": fingerprint(value)}
    if path.exists():
        if read_frozen_json(path) != sealed:
            raise ValueError(f"Frozen identity changed: {path}")
        return sealed
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(sealed, handle, indent=2, sort_keys=True, allow_nan=False)
    return sealed


def read_frozen_json(path: Path, expected: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    observed = value.pop("fingerprint", None)
    if observed != fingerprint(value) or (expected is not None and observed != expected):
        raise ValueError(f"Frozen identity mismatch: {path}")
    return {**value, "fingerprint": observed}
