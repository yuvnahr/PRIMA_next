"""Retrieval optimization runner tests."""

from evaluation.runners.retrieval_optimization_runner import RetrievalOptimizationRunner


def test_optimization_results_compare_baseline_and_best() -> None:
    runner = RetrievalOptimizationRunner()
    records = runner.semantic_runner._load_gold()[:30]
    repository = runner.semantic_runner._build_repository(records)
    graph_repository = runner.semantic_runner._build_graph(records)
    baseline = runner._trace_with_weights(
        records,
        repository,
        graph_repository,
        {"dense": 0.4, "sparse": 0.3, "temporal": 0.15, "graph": 0.15},
    )
    optimized = runner._trace_with_weights(
        records,
        repository,
        graph_repository,
        {"dense": 0.15, "sparse": 0.65, "temporal": 0.1, "graph": 0.1},
    )

    result = runner._optimization_results(baseline, optimized, optimized, optimized)

    assert "comparison" in result
    assert "recall_at_5_gain" in result["comparison"]
