"""Reflection metrics for runtime replay."""

from __future__ import annotations

from typing import Any


def trigger_frequency(records: list[dict[str, Any]]) -> int:
    """Count runtime turns that triggered reflection."""
    return sum(1 for record in records if record.get("reflection_triggered"))


def correction_frequency(records: list[dict[str, Any]]) -> int:
    """Count runtime turns that reported correction events."""
    return sum(int(record.get("correction_count", 0)) for record in records)
