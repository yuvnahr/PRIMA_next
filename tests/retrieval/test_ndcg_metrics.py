"""nDCG metric tests."""

from evaluation.metrics.retrieval_metrics import ndcg_at_k


def test_ndcg_is_one_for_ideal_binary_ranking() -> None:
    assert ndcg_at_k(["m1", "m2"], ["m1", "m2", "m9"], 5) == 1.0


def test_ndcg_discounts_late_relevant_items() -> None:
    late = ndcg_at_k(["m1"], ["m9", "m8", "m1"], 5)
    early = ndcg_at_k(["m1"], ["m1", "m8", "m9"], 5)

    assert 0.0 < late < early
