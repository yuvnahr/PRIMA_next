"""MRR metric tests."""

from evaluation.metrics.retrieval_metrics import reciprocal_rank, summarize_retrieval_metrics


def test_reciprocal_rank_uses_first_relevant_rank() -> None:
    assert reciprocal_rank(["m3"], ["m1", "m2", "m3"]) == 0.333333
    assert reciprocal_rank(["m3"], ["m3", "m2", "m1"]) == 1.0
    assert reciprocal_rank(["m3"], ["m1", "m2"]) == 0.0


def test_mrr_is_mean_reciprocal_rank() -> None:
    summary = summarize_retrieval_metrics(
        [
            {"expected_memory_ids": ["m1"], "retrieved_memory_ids": ["m1"], "latency_ms": 1.0},
            {"expected_memory_ids": ["m2"], "retrieved_memory_ids": ["m9", "m2"], "latency_ms": 1.0},
        ]
    )

    assert summary["mrr"] == 0.75
