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
