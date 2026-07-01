"""Tests for memory formation metrics."""

from __future__ import annotations

from evaluation.metrics.memory_formation import memory_formation_summary, memory_quality_comparison


def test_memory_formation_summary_counts_admissions() -> None:
    records = [
        {
            "query": "remember my allergy",
            "memory_created": True,
            "memory_importance_total": 0.8,
            "memory_importance_novelty": 0.9,
            "memory_importance_salience": 0.4,
        },
        {
            "query": "nice weather",
            "memory_created": False,
            "memory_importance_total": 0.2,
            "memory_importance_novelty": 0.1,
            "memory_importance_salience": 0.1,
        },
    ]

    summary = memory_formation_summary(records)

    assert summary["memory_creation_rate"] == 0.5
    assert summary["memory_rejection_rate"] == 0.5
    assert summary["average_importance_score"] == 0.5


def test_memory_quality_comparison_preserves_admitted_memory_retrieval() -> None:
    records = [
        {"query": "remember my allergy to peanuts", "memory_created": True, "memory_importance_total": 0.8},
        {"query": "casual weather note", "memory_created": False, "memory_importance_total": 0.2},
    ]

    comparison = memory_quality_comparison(records, top_k=5)

    assert comparison["benchmark_queries"] == 1
    assert comparison["recall_drop"] == 0.0
