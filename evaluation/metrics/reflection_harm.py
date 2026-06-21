"""Reflection harm analysis helpers."""

from __future__ import annotations

from typing import Any

from evaluation.metrics.reflection_accuracy import accuracy_records


def reflection_harm_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize when reflection improves or harms correctness."""
    benchmark_records = accuracy_records(records)
    total = len(benchmark_records)
    improvement_count = 0
    harm_count = 0

    for record in benchmark_records:
        correct_before = bool(record.get("correct_before", False))
        correct_after = bool(record.get("correct_after", False))
        if not correct_before and correct_after:
            improvement_count += 1
        elif correct_before and not correct_after:
            harm_count += 1

    improvement_rate = round(improvement_count / total, 6) if total else 0.0
    harm_rate = round(harm_count / total, 6) if total else 0.0
    return {
        "improvement_count": improvement_count,
        "harm_count": harm_count,
        "improvement_rate": improvement_rate,
        "harm_rate": harm_rate,
        "net_gain": round(improvement_rate - harm_rate, 6),
    }
