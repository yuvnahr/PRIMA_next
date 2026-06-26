"""Fusion optimizer tests."""

from evaluation.runners.retrieval_optimization_runner import RetrievalOptimizationRunner


def test_fusion_weight_search_selects_metrics() -> None:
    runner = RetrievalOptimizationRunner()
    records = runner.semantic_runner._load_gold()
    repository = runner.semantic_runner._build_repository(records)
    graph_repository = runner.semantic_runner._build_graph(records)

    result = runner._fusion_weight_search(records[:24], repository, graph_repository)

    assert "best_weights" in result
    assert "best_metrics" in result
    assert result["best_metrics"]["recall_at_5"] >= 0.0
