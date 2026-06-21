"""Retrieval benchmark runner tests."""

import json
from pathlib import Path

from evaluation.runners.retrieval_benchmark_runner import RetrievalBenchmarkRunner


def test_retrieval_benchmark_generates_trace_and_failure_files(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.json"
    records = []
    for index in range(100):
        note_id = f"bench_mem_{index:03d}"
        text = f"memory {index} about roses compost seedlings and personal gardening routine"
        records.append(
            {
                "query": text,
                "expected_memory_ids": [note_id],
                "memory_id": note_id,
                "memory_text": text,
                "memory_type": "semantic" if index % 2 else "episodic",
                "category": "semantic_memories" if index % 2 else "episodic_memories",
                "manually_verified": True,
            }
        )
    gold_path.write_text(json.dumps(records), encoding="utf-8")
    results_path = tmp_path / "results.json"
    trace_path = tmp_path / "trace.json"
    failures_path = tmp_path / "failures.json"

    report = RetrievalBenchmarkRunner(
        gold_path=gold_path,
        results_path=results_path,
        trace_path=trace_path,
        ablation_path=tmp_path / "ablation.json",
        failures_path=failures_path,
    ).run()

    assert report["dataset_size"] == 100
    assert results_path.exists()
    assert trace_path.exists()
    assert failures_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert len(trace) == 100
    assert {"query", "expected_memory_ids", "retrieved_memory_ids", "rank_positions", "hit"} <= set(trace[0])
