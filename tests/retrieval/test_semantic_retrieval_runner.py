"""Semantic retrieval runner tests."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.runners.retrieval_semantic_runner import SemanticRetrievalRunner


def _semantic_gold(path: Path) -> None:
    categories = ("preference", "identity", "temporal", "emotional", "multihop", "graph")
    records = []
    for index in range(150):
        category = categories[index % len(categories)]
        memory_id = f"semantic_mem_{index:03d}"
        linked_id = f"semantic_mem_{(index + 1) % 150:03d}"
        records.append(
            {
                "query": f"Which semantic benchmark clue concerns item {index} in category {category}?",
                "expected_memory_ids": [memory_id],
                "memory_id": memory_id,
                "category": category,
                "memory_type": "emotional" if category == "emotional" else "semantic",
                "memory_text": f"Stored benchmark memory {index} documents a {category} retrieval case.",
                "timestamp_offset_hours": index,
                "graph_links": [linked_id] if category in {"graph", "multihop"} else [],
                "manually_verified": True,
            }
        )
    path.write_text(json.dumps(records), encoding="utf-8")


def test_semantic_runner_writes_required_outputs(tmp_path: Path) -> None:
    gold_path = tmp_path / "retrieval_gold_v2.json"
    _semantic_gold(gold_path)

    report = SemanticRetrievalRunner(
        gold_path=gold_path,
        report_path=tmp_path / "retrieval_semantic_results.json",
        metrics_path=tmp_path / "retrieval_semantic_metrics.json",
        category_path=tmp_path / "retrieval_category_metrics.json",
        ablation_path=tmp_path / "retrieval_ablation_v2.json",
        contribution_path=tmp_path / "retrieval_component_contribution.json",
        failures_path=tmp_path / "retrieval_failures_v2.json",
        hard_queries_path=tmp_path / "hard_queries.json",
        trace_path=tmp_path / "retrieval_semantic_trace.json",
    ).run()

    assert report["dataset_size"] == 150
    assert set(report["category_metrics"]) == {"preference", "identity", "temporal", "emotional", "multihop", "graph"}
    assert (tmp_path / "retrieval_semantic_metrics.json").exists()
    assert (tmp_path / "retrieval_category_metrics.json").exists()
    assert (tmp_path / "retrieval_ablation_v2.json").exists()
    assert (tmp_path / "retrieval_component_contribution.json").exists()
    assert (tmp_path / "retrieval_failures_v2.json").exists()
    assert (tmp_path / "hard_queries.json").exists()
