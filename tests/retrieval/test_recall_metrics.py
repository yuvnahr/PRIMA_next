"""Recall and hit-rate metric tests."""

from evaluation.metrics.retrieval_metrics import hit_rate, recall_at_k, summarize_retrieval_metrics


def test_recall_at_k_uses_expected_relevant_set() -> None:
    assert recall_at_k(["m1", "m2"], ["m9", "m1", "m8"], 1) == 0.0
    assert recall_at_k(["m1", "m2"], ["m9", "m1", "m8"], 2) == 0.5
    assert recall_at_k(["m1", "m2"], ["m2", "m1", "m8"], 2) == 1.0


def test_hit_rate_is_binary_per_query() -> None:
    assert hit_rate(["m1"], ["m9", "m1"]) == 1.0
    assert hit_rate(["m1"], ["m9", "m8"]) == 0.0


def test_summary_includes_recall_levels() -> None:
    summary = summarize_retrieval_metrics(
        [
            {
                "expected_memory_ids": ["m1"],
                "retrieved_memory_ids": ["m1", "m2"],
                "latency_ms": 4.0,
            },
            {
                "expected_memory_ids": ["m3"],
                "retrieved_memory_ids": ["m1", "m2", "m3"],
                "latency_ms": 6.0,
            },
        ]
    )

    assert summary["recall_at_1"] == 0.5
    assert summary["recall_at_3"] == 1.0
    assert summary["average_retrieval_latency"] == 5.0
