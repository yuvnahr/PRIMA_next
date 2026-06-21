"""Reflection accuracy benchmark helpers."""

from __future__ import annotations

from typing import Any


def normalize_text(value: Any) -> str:
    """Normalize text for exact-match correctness checks."""
    return " ".join(str(value).strip().lower().split())


def accuracy_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only samples with ground-truth and before/after predictions."""
    benchmark_records: list[dict[str, Any]] = []
    for record in records:
        if "ground_truth" not in record and "ground_truth_emotion" not in record:
            continue
        if "prediction_before_reflection" not in record and "before_prediction" not in record:
            continue
        if "prediction_after_reflection" not in record and "after_prediction" not in record:
            continue

        ground_truth = str(record.get("ground_truth", record.get("ground_truth_emotion", "")))
        prediction_before = str(record.get("prediction_before_reflection", record.get("before_prediction", "")))
        prediction_after = str(record.get("prediction_after_reflection", record.get("after_prediction", "")))
        correct_before = normalize_text(prediction_before) == normalize_text(ground_truth)
        correct_after = normalize_text(prediction_after) == normalize_text(ground_truth)
        benchmark_records.append(
            {
                "query": str(record.get("query", "")),
                "ground_truth": ground_truth,
                "prediction_before_reflection": prediction_before,
                "prediction_after_reflection": prediction_after,
                "correct_before": correct_before,
                "correct_after": correct_after,
            }
        )
    return benchmark_records


def correct_before_count(records: list[dict[str, Any]]) -> int:
    """Count samples answered correctly before reflection."""
    return sum(1 for record in accuracy_records(records) if record["correct_before"])


def correct_after_count(records: list[dict[str, Any]]) -> int:
    """Count samples answered correctly after reflection."""
    return sum(1 for record in accuracy_records(records) if record["correct_after"])


def total_samples(records: list[dict[str, Any]]) -> int:
    """Count benchmarkable samples."""
    return len(accuracy_records(records))


def accuracy_before(records: list[dict[str, Any]]) -> float:
    """Compute the pre-reflection accuracy."""
    benchmark_records = accuracy_records(records)
    total = len(benchmark_records)
    return round(correct_before_count(records) / total, 6) if total else 0.0


def accuracy_after(records: list[dict[str, Any]]) -> float:
    """Compute the post-reflection accuracy."""
    benchmark_records = accuracy_records(records)
    total = len(benchmark_records)
    return round(correct_after_count(records) / total, 6) if total else 0.0


def accuracy_gain(records: list[dict[str, Any]]) -> float:
    """Compute the accuracy delta from reflection."""
    return round(accuracy_after(records) - accuracy_before(records), 6)


def reflection_accuracy_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the benchmark in a JSON-serializable payload."""
    benchmark_records = accuracy_records(records)
    total = len(benchmark_records)
    return {
        "total_samples": total,
        "correct_before": correct_before_count(records),
        "correct_after": correct_after_count(records),
        "accuracy_before": accuracy_before(records),
        "accuracy_after": accuracy_after(records),
        "accuracy_gain": accuracy_gain(records),
        "samples": benchmark_records,
    }


def print_reflection_accuracy_report(records: list[dict[str, Any]]) -> None:
    """Print a compact benchmark report."""
    summary = reflection_accuracy_summary(records)
    print("REFLECTION ACCURACY REPORT")
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Accuracy Before: {summary['accuracy_before']:.6f}")
    print(f"Accuracy After: {summary['accuracy_after']:.6f}")
    print(f"Accuracy Gain: {summary['accuracy_gain']:.6f}")
