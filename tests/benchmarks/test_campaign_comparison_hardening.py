from __future__ import annotations

import json
from datetime import UTC, datetime
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
from benchmarks.locomo.evaluate import paired_conversation_comparison


def _manifest(root: Path, selected: tuple[str, ...], *, dataset_hash: str = "dataset", benchmark: str = "GoEmotions") -> Path:
    manifest = BenchmarkManifest(
        campaign_id=root.name,
        benchmark=BenchmarkSpec(
            name=benchmark,
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
                "started_at": datetime.now(UTC).isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "total_ms": 1.0,
            },
            "diagnostics": {"locomo_record" if "conversation_id" in record else "goemotions_record": record},
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


def test_locomo_reporting_shares_cluster_intervals_and_names_the_scoring_rules(tmp_path):
    selected = ("a1", "a2", "b", "failure", "abstention")
    left = _manifest(tmp_path / "left", selected, benchmark="LoCoMo")
    right = _manifest(tmp_path / "right", selected, benchmark="LoCoMo")
    rows_a, rows_b = [], []
    for case_id in selected:
        gold = "apple, peach"
        a = {"case_id": case_id, "conversation_id": case_id[0], "category": 1,
             "question": "Which fruits?", "expected_answer": gold, "prediction": "apple",
             "reference_metadata": {"answer": gold}}
        b = {**a, "prediction": "apple, peach"}
        if case_id == "b":
            a["prediction"], b["prediction"] = b["prediction"], a["prediction"]
        if case_id == "failure":
            b["execution_failed"] = True
        if case_id == "abstention":
            b["outcome"] = "abstained"
        rows_a.append(a)
        rows_b.append(b)
        _checkpoint(left.parent, case_id, a)
    for b in reversed(rows_b):
        _checkpoint(right.parent, b["case_id"], b)

    def campaign(metric):
        spec = ComparisonConfig(id="locomo", left="left", right="right", metric=metric, bootstrap_samples=200)
        return compare(spec, left, right, {}, {}, "revision", 512, 512)

    native = paired_conversation_comparison(rows_a, rows_b, samples=200, seed=7)
    official = campaign("answerable_official_f1")
    assert official["bootstrap_95_ci"] == native["metrics"]["answerable_official_f1"]["bootstrap_95_ci"] == [-.5, .5]
    assert official["independent_units"] == 2 and official["paired_case_ids"] == list(selected[:3])
    assert official["exclusions_by_arm"]["right"]["failed"] == 1
    assert official["answerable_population"]["right"] == {
        "selected": 5, "answered": 3, "answered_official_f1": 2.5 / 3,
        "coverage": .6, "all_selected_official_f1": .5,
    }
    assert campaign("f1")["metric_definition"] == "legacy_squad_f1"
    assert campaign("f1")["left"] != campaign("official_locomo_f1")["left"]

    checkpoint = right.parent / "checkpoints" / "records.jsonl"
    checkpoint.write_text(checkpoint.read_text().replace('"case_id": "a1", "conversation_id"', '"case_id": "wrong", "conversation_id"'))
    with pytest.raises(ValueError, match="native record ID"):
        campaign("answerable_official_f1")
