"""Tests for reflection harm analysis helpers."""

from __future__ import annotations

from evaluation.metrics.reflection_harm import reflection_harm_summary


def test_reflection_harm_summary_counts_harm_and_improvement() -> None:
    records = [
        {
            "query": "Q1",
            "ground_truth": "Richard Nixon",
            "prediction_before_reflection": "Abraham Lincoln",
            "prediction_after_reflection": "Richard Nixon",
        },
        {
            "query": "Q2",
            "ground_truth": "Richard Nixon",
            "prediction_before_reflection": "Richard Nixon",
            "prediction_after_reflection": "Abraham Lincoln",
        },
        {
            "query": "Q3",
            "ground_truth": "Richard Nixon",
            "prediction_before_reflection": "Richard Nixon",
            "prediction_after_reflection": "Richard Nixon",
        },
        {
            "query": "Q4",
            "ground_truth": "Richard Nixon",
            "prediction_before_reflection": "Abraham Lincoln",
            "prediction_after_reflection": "Abraham Lincoln",
        },
    ]

    summary = reflection_harm_summary(records)

    assert summary == {
        "improvement_count": 1,
        "harm_count": 1,
        "improvement_rate": 0.25,
        "harm_rate": 0.25,
        "net_gain": 0.0,
    }
