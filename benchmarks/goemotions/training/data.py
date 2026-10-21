"""Split-safe official GoEmotions data access for training workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.goemotions.dataset import GoEmotionsExample, load_examples, load_labels


def load_split(data_dir: Path, split: str) -> list[GoEmotionsExample]:
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    return load_examples(data_dir / f"{split}.tsv", data_dir / "emotions.txt")


def validate_splits(data_dir: Path) -> dict[str, object]:
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
    result = {
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
