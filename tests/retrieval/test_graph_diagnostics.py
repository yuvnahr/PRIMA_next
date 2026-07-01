"""Graph diagnostics tests."""

from evaluation.runners.retrieval_optimization_runner import RetrievalOptimizationRunner


def test_graph_diagnostics_reports_graph_queries() -> None:
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

    diagnostics = runner._graph_diagnostics(records, graph_repository, trace)

    assert diagnostics["graph_query_count"] > 0
    assert "graph_recall" in diagnostics
    assert diagnostics["diagnostics"]
