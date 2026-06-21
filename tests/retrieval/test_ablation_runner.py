"""Retrieval ablation runner tests."""

import json
from pathlib import Path

from evaluation.runners.retrieval_benchmark_runner import RetrievalBenchmarkRunner


def _write_gold(path: Path) -> None:
    records = []
    for index in range(100):
        note_id = f"test_mem_{index:03d}"
        text = f"unique retrieval benchmark memory {index} tomato basil greenhouse project"
        records.append(
            {
                "query": text,
                "expected_memory_ids": [note_id],
                "memory_id": note_id,
                "memory_text": text,
                "memory_type": "episodic",
                "category": "episodic_memories",
                "manually_verified": True,
            }
        )
    path.write_text(json.dumps(records), encoding="utf-8")


def test_ablation_runner_writes_all_configurations(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.json"
    _write_gold(gold_path)
    runner = RetrievalBenchmarkRunner(
        gold_path=gold_path,
        results_path=tmp_path / "results.json",
        trace_path=tmp_path / "trace.json",
        ablation_path=tmp_path / "ablation.json",
        failures_path=tmp_path / "failures.json",
    )

    result = runner.run()

    assert set(result["ablation"]) == {
        "dense_only",
        "dense_sparse",
        "dense_sparse_temporal",
        "dense_sparse_temporal_graph",
    }
    assert (tmp_path / "ablation.json").exists()
    assert result["ablation"]["dense_sparse_temporal_graph"]["recall_at_5"] >= result["ablation"]["dense_only"]["recall_at_5"]
