"""Confidence calibration audit helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from evaluation.metrics.reflection_accuracy import normalize_text

BROKERS = tuple(round(index / 10, 1) for index in range(10))


def _bucket_label(lower: float) -> str:
    upper = round(min(1.0, lower + 0.1), 1)
    return f"{lower:.1f}-{upper:.1f}"


def _bucket_lower(confidence: float) -> float:
    value = max(0.0, min(1.0, confidence))
    if value >= 1.0:
        return 0.9
    return round((int(value * 10)) / 10, 1)


def calibration_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build bucketed calibration records from the benchmarkable samples."""
    buckets: OrderedDict[float, list[dict[str, Any]]] = OrderedDict((lower, []) for lower in BROKERS)
    for record in records:
        if "ground_truth" not in record and "ground_truth_emotion" not in record:
            continue
        if "prediction_after_reflection" not in record and "after_prediction" not in record:
            continue
        if "confidence" not in record:
            continue

        ground_truth = str(record.get("ground_truth", record.get("ground_truth_emotion", "")))
        prediction_after = str(record.get("prediction_after_reflection", record.get("after_prediction", "")))
        correct_after = normalize_text(prediction_after) == normalize_text(ground_truth)
        confidence = float(record.get("confidence", 0.0))
        bucket = _bucket_lower(confidence)
        buckets[bucket].append(
            {
                "query": str(record.get("query", "")),
                "confidence": confidence,
                "correct_after": correct_after,
            }
        )

    calibration: list[dict[str, Any]] = []
    for lower, bucket_records in buckets.items():
        sample_count = len(bucket_records)
        average_confidence = round(
            sum(max(0.0, min(1.0, float(record.get("confidence", 0.0)))) for record in bucket_records) / sample_count,
            6,
        ) if sample_count else 0.0
        actual_accuracy = round(sum(1 for record in bucket_records if bool(record.get("correct_after", False))) / sample_count, 6) if sample_count else 0.0
        calibration.append(
            {
                "bucket": _bucket_label(lower),
                "sample_count": sample_count,
                "average_confidence": average_confidence,
                "actual_accuracy": actual_accuracy,
            }
        )
    return calibration


def confidence_calibration_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize calibration quality for JSON output."""
    buckets = calibration_records(records)
    total_samples = sum(bucket["sample_count"] for bucket in buckets)
    weighted_accuracy = (
        round(
            sum(bucket["sample_count"] * bucket["actual_accuracy"] for bucket in buckets) / total_samples,
            6,
        )
        if total_samples
        else 0.0
    )
    average_confidence = (
        round(
            sum(bucket["sample_count"] * bucket["average_confidence"] for bucket in buckets) / total_samples,
            6,
        )
        if total_samples
        else 0.0
    )
    calibration_gap = round(weighted_accuracy - average_confidence, 6)
    return {
        "total_samples": total_samples,
        "average_confidence": average_confidence,
        "actual_accuracy": weighted_accuracy,
        "calibration_gap": calibration_gap,
        "buckets": buckets,
    }


def print_confidence_calibration_report(records: list[dict[str, Any]]) -> None:
    """Print a concise calibration summary."""
    summary = confidence_calibration_summary(records)
    print("CONFIDENCE CALIBRATION AUDIT")
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Average Confidence: {summary['average_confidence']:.6f}")
    print(f"Actual Accuracy: {summary['actual_accuracy']:.6f}")
    print(f"Calibration Gap: {summary['calibration_gap']:.6f}")
