from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from affect.taxonomies.goemotions import LABELS
from benchmarks.campaign.comparison import compare
from benchmarks.campaign.config import ComparisonConfig
from benchmarks.common.contracts import (
    BenchmarkManifest,
    BenchmarkMode,
    BenchmarkSpec,
    RunStatus,
)
from benchmarks.goemotions.metrics import evaluate


def _manifest(root: Path, selected: tuple[str, ...], *, dataset_hash: str = "dataset") -> Path:
    manifest = BenchmarkManifest(
        campaign_id=root.name,
        benchmark=BenchmarkSpec(
            name="GoEmotions",
            version="1.0",
            mode=BenchmarkMode.CLASSIFICATION,
            dataset_name="test.tsv",
        ),
        source_fingerprint="source",
        dataset_hash=dataset_hash,
        selected_ids=selected,
        provider="fake",
        model="fixture",
        model_revision="revision",
        generation_config={"seed": 7, "retries": 1},
        benchmark_config={"split": "test", "dataset_scope": "sample", "system": root.name},
        runtime_profile="affect_only",
        active_capabilities={"headline_eligible": True},
        repository_mode="benchmark:in_memory",
        prompt_hashes={},
        seed=7,
        dependencies={},
        hardware={},
        status=RunStatus.COMPLETE,
    )
    root.mkdir(parents=True)
    path = root / "manifest.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    return path


def _checkpoint(root: Path, case_id: str, record: dict | None, *, status: str = "complete") -> None:
    path = root / "checkpoints" / "records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case_id,
        "status": status,
        "prediction": None if record is None else {
            "case_id": case_id,
            "mode": "classification",
            "prediction": record.get("predicted_labels", []),
            "timing": {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "total_ms": 1.0,
            },
            "diagnostics": {"goemotions_record": record},
        },
        "failure": {"message": "failed"} if status != "complete" else None,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\n")


def test_comparison_recomputes_macro_f1_and_excludes_unscored_cases(tmp_path: Path) -> None:
    selected = ("a", "b", "failed", "parse")
    left = _manifest(tmp_path / "left", selected)
    right = _manifest(tmp_path / "right", selected)
    left_records = [
        {"gold_labels": ["joy"], "predicted_labels": ["joy", "anger"]},
        {"gold_labels": ["anger"], "predicted_labels": []},
    ]
    right_records = [
        {"gold_labels": ["joy"], "predicted_labels": ["joy"]},
        {"gold_labels": ["anger"], "predicted_labels": ["anger"]},
    ]
    for case_id, left_record, right_record in zip(selected[:2], left_records, right_records, strict=True):
        _checkpoint(left.parent, case_id, left_record)
        _checkpoint(right.parent, case_id, right_record)
    _checkpoint(left.parent, "failed", None, status="failed")
    _checkpoint(right.parent, "failed", None, status="failed")
    parse_record = {"gold_labels": ["joy"], "predicted_labels": [], "parse_error": "malformed"}
    _checkpoint(left.parent, "parse", parse_record)
    _checkpoint(right.parent, "parse", parse_record)

    result = compare(
        ComparisonConfig(id="native", left="left", right="right", metric="metrics.macro_f1", bootstrap_samples=20),
        left,
        right,
        {},
        {},
        "revision",
        512,
        512,
    )

    expected_left = evaluate(
        [frozenset(row["gold_labels"]) for row in left_records],
        [frozenset(row["predicted_labels"]) for row in left_records],
        list(LABELS),
    )["macro_f1"]
    assert result["left"] == pytest.approx(expected_left)
    assert result["right"] > result["left"]
    assert result["paired_scored"] == 2
    assert result["excluded"] == {"failed": 1, "parse_failed": 1, "unscored": 0}


def test_comparison_refuses_dataset_hash_mismatch(tmp_path: Path) -> None:
    left = _manifest(tmp_path / "left", ("a",), dataset_hash="left")
    right = _manifest(tmp_path / "right", ("a",), dataset_hash="right")
    with pytest.raises(ValueError, match="dataset_hash"):
        compare(
            ComparisonConfig(id="invalid", left="left", right="right", metric="metrics.macro_f1"),
            left,
            right,
            {},
            {},
            "revision",
            512,
            512,
        )
