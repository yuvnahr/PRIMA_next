"""Temporal diagnostics tests."""

from evaluation.runners.retrieval_optimization_runner import RetrievalOptimizationRunner


def test_temporal_diagnostics_is_category_scoped() -> None:
    runner = RetrievalOptimizationRunner()
    records = runner.semantic_runner._load_gold()
    repository = runner.semantic_runner._build_repository(records)
    graph_repository = runner.semantic_runner._build_graph(records)
    trace = runner._trace_with_weights(
        records,
        repository,
        graph_repository,
        {"dense": 0.15, "sparse": 0.65, "temporal": 0.1, "graph": 0.1},
    )

    diagnostics = runner._temporal_diagnostics(records, trace)

    assert diagnostics["temporal_query_count"] == 30
    assert "temporal_metrics" in diagnostics
