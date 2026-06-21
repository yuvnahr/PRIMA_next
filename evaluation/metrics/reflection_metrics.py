"""Reflection metrics for runtime replay."""

from __future__ import annotations

from typing import Any


def trigger_frequency(records: list[dict[str, Any]]) -> int:
    """Count runtime turns that triggered reflection."""
    return sum(1 for record in records if record.get("reflection_triggered"))


def correction_frequency(records: list[dict[str, Any]]) -> int:
    """Count runtime turns that reported correction events."""
    return sum(int(record.get("correction_count", 0)) for record in records)


def trigger_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count reflection trigger reasons by category."""
    counts = {
        "low_confidence": 0,
        "tool_failure": 0,
        "contradiction": 0,
        "hallucination": 0,
        "constraint_violation": 0,
    }
    for record in records:
        if not record.get("reflection_triggered"):
            continue
        categories_seen: set[str] = set()
        for reason in record.get("reflection_reasons", []):
            if not isinstance(reason, dict) or not reason.get("triggered"):
                continue
            reason_name = str(reason.get("reason", ""))
            category = "low_confidence" if reason_name == "retrieval_ambiguity" else reason_name
            if category in counts:
                categories_seen.add(category)
        for category in categories_seen:
            counts[category] += 1
    return counts


def reflection_rate(records: list[dict[str, Any]]) -> float:
    """Return the proportion of turns that triggered reflection."""
    return round(trigger_frequency(records) / len(records), 6) if records else 0.0


def correction_rate(records: list[dict[str, Any]]) -> float:
    """Return the proportion of turns with useful reflection corrections."""
    return round(correction_frequency(records) / len(records), 6) if records else 0.0


def utility_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize before/after confidence lift from reflection."""
    utilities = [float(record.get("reflection_utility_score", 0.0)) for record in records]
    triggered = [float(record.get("reflection_utility_score", 0.0)) for record in records if record.get("reflection_triggered")]
    return {
        "mean_utility": round(sum(utilities) / len(utilities), 6) if utilities else 0.0,
        "mean_triggered_utility": round(sum(triggered) / len(triggered), 6) if triggered else 0.0,
        "max_utility": round(max(utilities), 6) if utilities else 0.0,
    }
