"""Tests for reflection accuracy benchmark helpers."""

from __future__ import annotations

from evaluation.metrics.reflection_accuracy import accuracy_after, accuracy_before, accuracy_gain, reflection_accuracy_summary


def test_reflection_accuracy_summary_counts_correct_predictions() -> None:
    records = [
        {
            "query": "Q1",
            "ground_truth": "Richard Nixon",
            "prediction_before_reflection": "Richard Nixon",
            "prediction_after_reflection": "Richard Nixon",
        },
        {
            "query": "Q2",
            "ground_truth": "Richard Nixon",
            "prediction_before_reflection": "Abraham Lincoln",
            "prediction_after_reflection": "Richard Nixon",
        },
    ]

    summary = reflection_accuracy_summary(records)

    assert summary["total_samples"] == 2
    assert summary["correct_before"] == 1
    assert summary["correct_after"] == 2
    assert summary["accuracy_before"] == 0.5
    assert summary["accuracy_after"] == 1.0
    assert summary["accuracy_gain"] == 0.5
    assert accuracy_before(records) == 0.5
    assert accuracy_after(records) == 1.0
    assert accuracy_gain(records) == 0.5
