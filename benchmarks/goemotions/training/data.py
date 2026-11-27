"""Split-safe official GoEmotions data access for training workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from affect.classifiers.calibration_artifact import encoder_identity
from affect.taxonomies.goemotions import LABELS
from benchmarks.goemotions.dataset import GoEmotionsExample, load_examples, load_labels
from config.frozen_json import fingerprint, read_frozen_json


def load_split(data_dir: Path, split: str) -> list[GoEmotionsExample]:
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    return load_examples(data_dir / f"{split}.tsv", data_dir / "emotions.txt")


def validate_splits(data_dir: Path) -> dict[str, Any]:
    labels = load_labels(data_dir / "emotions.txt")
    splits = {name: load_split(data_dir, name) for name in ("train", "dev", "test")}
    ids = {name: {row.example_id for row in rows} for name, rows in splits.items()}
    duplicates = {name: len(rows) - len(ids[name]) for name, rows in splits.items()}
    if any(duplicates.values()):
        raise ValueError(f"Duplicate GoEmotions IDs detected within splits: {duplicates}")
    overlap = {
        f"{left}_{right}": sorted(ids[left] & ids[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    if any(overlap.values()):
        raise ValueError("GoEmotions split ID leakage detected.")
    texts = {name: {row.text for row in rows} for name, rows in splits.items()}
    result: dict[str, Any] = {
        "labels": labels,
        "label_sha256": hashlib.sha256((data_dir / "emotions.txt").read_bytes()).hexdigest(),
        "exact_text_overlap": {
            f"{left}_{right}": len(texts[left] & texts[right])
            for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
        },
        "repeated_text_rows": {name: len(rows) - len(texts[name]) for name, rows in splits.items()},
        "counts": {name: len(rows) for name, rows in splits.items()},
        "hashes": {name: hashlib.sha256((data_dir / f"{name}.tsv").read_bytes()).hexdigest() for name in splits},
        "overlap": overlap,
        "duplicates": duplicates,
    }
    result["fingerprint"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def multi_hot(example: GoEmotionsExample, labels: list[str]) -> list[float]:
    return [1.0 if label in example.labels else 0.0 for label in labels]


def load_encoder_dev_predictions(
    data_dir: Path, path: Path, *, model_id: str, revision: str, max_length: int,
    representation: str, probability_transform: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[GoEmotionsExample], dict[str, Any]]:
    dev = load_split(data_dir, "dev")
    if tuple(load_labels(data_dir / "emotions.txt")) != LABELS:
        raise ValueError("Development labels must match the official ordered taxonomy")
    ids = [row.example_id for row in dev]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Development split requires unique example IDs")
    identity = encoder_identity(model_id, revision, max_length,
        hashlib.sha256((data_dir / "dev.tsv").read_bytes()).hexdigest(), fingerprint(ids))
    result = read_frozen_json(path)
    if (result.get("identity") != identity or result.get("example_ids") != ids
        or result.get("representation") != representation
        or result.get("probability_transform") != probability_transform):
        raise ValueError("Development predictions have mismatched model, preprocessing, data, order or transform")
    rows = result.get("rows")
    if not isinstance(rows, list) or len(rows) != len(dev):
        raise ValueError("Development predictions must cover each ordered example exactly once")
    return identity, dev, result
