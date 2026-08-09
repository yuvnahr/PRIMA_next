"""Tests for confidence calibration audit helpers."""

from __future__ import annotations

from evaluation.metrics.confidence_calibration import (
    calibration_records,
    confidence_calibration_summary,
)


def test_confidence_calibration_summary_buckets_confidence() -> None:
    records = [
        {
            "query": "Q1",
            "ground_truth": "Richard Nixon",
            "prediction_before_reflection": "Abraham Lincoln",
            "prediction_after_reflection": "Richard Nixon",
            "confidence": 0.05,
        },
        {
            "query": "Q2",
            "ground_truth": "Richard Nixon",
            "prediction_before_reflection": "Richard Nixon",
            "prediction_after_reflection": "Richard Nixon",
            "confidence": 0.45,
        },
        {
            "query": "Q3",
            "ground_truth": "Richard Nixon",
            "prediction_before_reflection": "Richard Nixon",
            "prediction_after_reflection": "Abraham Lincoln",
            "confidence": 0.95,
        },
    ]

    summary = confidence_calibration_summary(records)
    buckets = calibration_records(records)

    assert summary["total_samples"] == 3
    assert summary["average_confidence"] == 0.483333
    assert summary["actual_accuracy"] == 0.666667
    assert summary["calibration_gap"] == 0.183334
    assert buckets[0] == {
        "bucket": "0.0-0.1",
        "sample_count": 1,
        "average_confidence": 0.05,
        "actual_accuracy": 1.0,
    }
    assert buckets[4] == {
        "bucket": "0.4-0.5",
        "sample_count": 1,
        "average_confidence": 0.45,
        "actual_accuracy": 1.0,
    }
    assert buckets[9] == {
        "bucket": "0.9-1.0",
        "sample_count": 1,
        "average_confidence": 0.95,
        "actual_accuracy": 0.0,
    }
