"""Memory metrics for runtime replay."""

from __future__ import annotations

from typing import Any


def retrieval_hit_counts(records: list[dict[str, Any]]) -> list[int]:
    """Return retrieval counts for each replayed query."""
    return [int(record.get("retrieval_count", 0)) for record in records]


def memory_growth(records: list[dict[str, Any]]) -> int:
    """Count records that created at least one memory."""
    return sum(1 for record in records if record.get("memory_created"))
