"""Dataset runner integration tests."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.runners.dataset_runner import DatasetRunner
from runtime import PrimaRuntime


def test_dataset_replay_executes_and_generates_files(tmp_path: Path) -> None:
    dataset_path = tmp_path / "inputs.json"
    results_path = tmp_path / "runtime_results.json"
    metrics_path = tmp_path / "runtime_metrics.json"
    reflection_log_path = tmp_path / "reflection_log.json"
    reflection_accuracy_path = tmp_path / "reflection_accuracy.json"
    reflection_change_report_path = tmp_path / "reflection_change_report.json"
    confidence_calibration_path = tmp_path / "confidence_calibration.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "query": "Who was Milhouse named after?",
                    "ground_truth": "Richard Nixon",
                    "prediction_before_reflection": "Richard Nixon",
                    "prediction_after_reflection": "Richard Nixon",
                },
                {
                    "query": "Who was Milhouse named after?",
                    "ground_truth": "Richard Nixon",
                    "prediction_before_reflection": "Abraham Lincoln",
                    "prediction_after_reflection": "Richard Nixon",
                },
            ]
        ),
        encoding="utf-8",
    )

    records = DatasetRunner(
        runtime=PrimaRuntime(),
        dataset_path=dataset_path,
        results_path=results_path,
        metrics_path=metrics_path,
        reflection_log_path=reflection_log_path,
        reflection_change_report_path=reflection_change_report_path,
        reflection_accuracy_path=reflection_accuracy_path,
        confidence_calibration_path=confidence_calibration_path,
    ).run()

    assert len(records) == 2
    assert results_path.exists()
    assert metrics_path.exists()
    assert reflection_log_path.exists()
    assert reflection_accuracy_path.exists()
    assert reflection_change_report_path.exists()
    assert confidence_calibration_path.exists()
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    assert len(saved) == 2
    assert all("retrieval_count" in record for record in saved)
    reflection_log = json.loads(reflection_log_path.read_text(encoding="utf-8"))
    assert all("before_prediction" in record for record in reflection_log)
    assert all("after_prediction" in record for record in reflection_log)
    assert all("changed" in record for record in reflection_log)
    change_report = json.loads(reflection_change_report_path.read_text(encoding="utf-8"))
    assert len(change_report) == 1
    assert change_report[0]["query"] == "Who was Milhouse named after?"
    assert change_report[0]["before_prediction"] == "Abraham Lincoln"
    assert change_report[0]["after_prediction"] == "Richard Nixon"
    assert change_report[0]["changed"] is True
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["reflection_accuracy"]["total_samples"] == 2
    assert metrics["reflection_accuracy"]["accuracy_before"] == 0.5
    assert metrics["reflection_accuracy"]["accuracy_after"] == 1.0
    assert metrics["reflection_accuracy"]["accuracy_gain"] == 0.5
    assert metrics["reflection_harm"] == {
        "improvement_count": 1,
        "harm_count": 0,
        "improvement_rate": 0.5,
        "harm_rate": 0.0,
        "net_gain": 0.5,
    }
    assert metrics["reflection"]["affect_trigger_count"] >= 0
    assert metrics["reflection"]["retrieval_trigger_count"] >= 0
    assert metrics["reflection"]["contradiction_trigger_count"] >= 0
    assert metrics["reflection"]["trigger_summary"] == {
        "affect_trigger_count": metrics["reflection"]["affect_trigger_count"],
        "retrieval_trigger_count": metrics["reflection"]["retrieval_trigger_count"],
        "contradiction_trigger_count": metrics["reflection"]["contradiction_trigger_count"],
    }
    assert metrics["confidence_calibration"]["total_samples"] == 2
    assert len(metrics["confidence_calibration"]["buckets"]) == 10
