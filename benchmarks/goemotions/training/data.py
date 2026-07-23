"""Split-safe official GoEmotions data access for training workflows."""

from __future__ import annotations

import hashlib
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
    overlap = {f"{left}_{right}": sorted(ids[left] & ids[right]) for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    if any(overlap.values()):
        raise ValueError("GoEmotions split ID leakage detected.")
    return {"labels": labels, "counts": {name: len(rows) for name, rows in splits.items()}, "hashes": {name: hashlib.sha256((data_dir / f"{name}.tsv").read_bytes()).hexdigest() for name in splits}, "overlap": overlap}


def multi_hot(example: GoEmotionsExample, labels: list[str]) -> list[float]:
    return [1.0 if label in example.labels else 0.0 for label in labels]
