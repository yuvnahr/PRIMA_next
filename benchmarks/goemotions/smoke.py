"""Fixed-manifest GoEmotions smoke comparison helpers."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from benchmarks.goemotions.dataset import DEFAULT_DATASET_PATH, load_examples


def create_sample_manifest(dataset_path: Path = DEFAULT_DATASET_PATH, output_path: Path = Path("evaluation/goemotions/smoke/sample_manifest.json"), seed: int = 13, size: int = 20) -> dict[str, Any]:
    """Select a deterministic, unedited test sample before system comparison."""
    examples = load_examples(dataset_path, dataset_path.with_name("emotions.txt"))
    if size > len(examples):
        raise ValueError("Smoke sample size exceeds test split size.")
    selected = random.Random(seed).sample(examples, size)  # nosec B311
    payload = {"seed": seed, "split": "test", "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(), "samples": [{"id": row.example_id, "labels": sorted(row.labels)} for row in selected]}
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def create_diagnostic_dev_manifest(
    dataset_path: Path = DEFAULT_DATASET_PATH.with_name("dev.tsv"),
    output_path: Path = Path("evaluation/goemotions/smoke/dev_diagnostic_manifest.json"),
) -> dict[str, Any]:
    """Select 20 deterministic development examples covering benchmark edge cases."""
    examples = load_examples(dataset_path, dataset_path.with_name("emotions.txt"))
    selected: list[tuple[str, Any]] = []
    used: set[str] = set()
    buckets = [
        ("neutral_combination", 5, lambda row: "neutral" in row.labels and len(row.labels) > 1),
        ("multilabel", 5, lambda row: "neutral" not in row.labels and len(row.labels) > 1),
        ("rare_grief", 1, lambda row: "grief" in row.labels),
        ("rare_nervousness", 1, lambda row: "nervousness" in row.labels),
        ("rare_pride", 1, lambda row: "pride" in row.labels),
        ("rare_relief", 1, lambda row: "relief" in row.labels),
        ("confused_anger_annoyance", 1, lambda row: bool(row.labels & {"anger", "annoyance"})),
        ("confused_fear_nervousness", 1, lambda row: bool(row.labels & {"fear", "nervousness"})),
        ("confused_sadness_disappointment", 1, lambda row: bool(row.labels & {"sadness", "disappointment"})),
        ("confused_approval_admiration", 1, lambda row: bool(row.labels & {"approval", "admiration"})),
        ("confused_confusion_curiosity", 1, lambda row: bool(row.labels & {"confusion", "curiosity"})),
        ("confused_surprise_realization", 1, lambda row: bool(row.labels & {"surprise", "realization"})),
    ]
    for category, count, matches in buckets:
        matches_for_bucket = [row for row in examples if row.example_id not in used and matches(row)][:count]
        if len(matches_for_bucket) != count:
            raise ValueError(f"Not enough development examples for {category}.")
        selected.extend((category, row) for row in matches_for_bucket)
        used.update(row.example_id for row in matches_for_bucket)
    payload = {
        "seed": 13,
        "split": "dev",
        "selection": "diagnostic_not_for_model_selection",
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "samples": [{"id": row.example_id, "labels": sorted(row.labels), "category": category} for category, row in selected],
    }
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(create_diagnostic_dev_manifest(), indent=2, sort_keys=True))
