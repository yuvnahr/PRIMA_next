"""System-level runtime metrics."""

from __future__ import annotations

from typing import Any


def summarize_system_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Compute aggregate runtime health metrics."""
    total = len(records)
    if total == 0:
        return {
            "average_latency": 0.0,
            "average_confidence": 0.0,
            "reflection_rate": 0.0,
            "memory_creation_rate": 0.0,
        }
    return {
        "average_latency": round(sum(float(record.get("latency_ms", 0.0)) for record in records) / total, 6),
        "average_confidence": round(sum(float(record.get("confidence", 0.0)) for record in records) / total, 6),
        "reflection_rate": round(sum(1 for record in records if record.get("reflection_triggered")) / total, 6),
        "memory_creation_rate": round(sum(1 for record in records if record.get("memory_created")) / total, 6),
    }
