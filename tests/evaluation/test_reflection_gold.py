"""Tests for the reflection gold benchmark runner path."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.runners.dataset_runner import DatasetRunner


def test_reflection_gold_dataset_generates_results(tmp_path: Path) -> None:
    dataset_path = tmp_path / "reflection_gold.json"
    results_path = tmp_path / "reflection_gold_results.json"
    metrics_path = tmp_path / "runtime_metrics.json"
    reflection_accuracy_path = tmp_path / "reflection_accuracy.json"
    outcomes_path = tmp_path / "reflection_outcomes.json"
    recovery_report_path = tmp_path / "reflection_recovery_report.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"query": "I got promoted today and could not stop smiling.", "ground_truth_emotion": "joy"},
                {"query": "My heart sank when I realized aphids had infested the leaves.", "ground_truth_emotion": "fear"},
            ]
        ),
        encoding="utf-8",
    )

    records = DatasetRunner(
        dataset_path=dataset_path,
        results_path=tmp_path / "runtime_results.json",
        metrics_path=metrics_path,
        reflection_accuracy_path=reflection_accuracy_path,
        reflection_gold_results_path=results_path,
        reflection_outcomes_path=outcomes_path,
        reflection_recovery_report_path=recovery_report_path,
    ).run()

    assert len(records) == 2
    assert results_path.exists()
    assert outcomes_path.exists()
    assert recovery_report_path.exists()
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    assert len(saved) == 2
    assert set(saved[0]) == {
        "query",
        "ground_truth",
        "before_prediction",
        "after_prediction",
        "reflection_candidate_prediction",
        "reflection_triggered",
        "reflection_accepted",
        "reflection_outcome",
        "correct_before",
        "correct_after",
    }
    recovery = json.loads(recovery_report_path.read_text(encoding="utf-8"))
    assert set(recovery) == {
        "total_samples",
        "reflection_trigger_count",
        "reflection_acceptance_count",
        "improvement_count",
        "harm_count",
        "reflection_trigger_rate",
        "reflection_acceptance_rate",
        "improvement_rate",
        "harm_rate",
        "net_gain",
    }
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["reflection_accuracy"]["total_samples"] == 2
    assert metrics["confidence_calibration"]["total_samples"] == 2
