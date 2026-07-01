"""Embedding diagnostics tests."""

from evaluation.runners.retrieval_optimization_runner import RetrievalOptimizationRunner


def test_embedding_diagnostics_reports_similarity_fields() -> None:
    runner = RetrievalOptimizationRunner()
    records = runner.semantic_runner._load_gold()[:12]
    repository = runner.semantic_runner._build_repository(records)
    graph_repository = runner.semantic_runner._build_graph(records)
    trace = runner._trace_with_weights(
        records,
        repository,
        graph_repository,
        {"dense": 0.15, "sparse": 0.65, "temporal": 0.1, "graph": 0.1},
    )

    diagnostics = runner._embedding_diagnostics(records, repository, trace)

    assert "nearest_neighbor_accuracy" in diagnostics
    assert {"expected_similarity", "top_retrieved_similarity", "embedding_margin"} <= set(diagnostics["diagnostics"][0])
