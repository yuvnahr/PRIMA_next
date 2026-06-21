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
    dataset_path.write_text(
        json.dumps(
            [
                {"raw_text": "I watered the seedlings before sunrise."},
                {"raw_text": "The wilting leaves made me anxious."},
            ]
        ),
        encoding="utf-8",
    )

    records = DatasetRunner(
        runtime=PrimaRuntime(),
        dataset_path=dataset_path,
        results_path=results_path,
        metrics_path=metrics_path,
    ).run()

    assert len(records) == 2
    assert results_path.exists()
    assert metrics_path.exists()
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    assert len(saved) == 2
    assert all("retrieval_count" in record for record in saved)
