"""Load the official GoEmotions TSV splits without copying the dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from affect.taxonomies.goemotions import LABELS

BENCHMARK_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = BENCHMARK_ROOT / "external" / "goemotions" / "data" / "test.tsv"
DEFAULT_LABEL_PATH = DEFAULT_DATASET_PATH.with_name("emotions.txt")


@dataclass(frozen=True)
class GoEmotionsExample:
    """One official GoEmotions TSV example."""

    text: str
    labels: frozenset[str]
    example_id: str


def load_labels(path: Path = DEFAULT_LABEL_PATH) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 28 or labels[-1] != "neutral":
        raise ValueError(f"Expected the official 28 GoEmotions labels in {path}.")
    return labels


def load_examples(dataset_path: Path = DEFAULT_DATASET_PATH, label_path: Path | None = None) -> list[GoEmotionsExample]:
    """Parse the official ``text, labels, id`` TSV or JSON format."""

    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"GoEmotions dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() == ".json":
        try:
            rows = json.loads(dataset_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid GoEmotions JSON in {dataset_path}") from exc
        if not isinstance(rows, list):
            raise ValueError(f"Expected a GoEmotions JSON array in {dataset_path}")
        examples = []
        for row_number, row in enumerate(rows, start=1):
            try:
                text, indices, example_id = row["text"], row["labels"], row["id"]
                if not isinstance(text, str) or not text.strip() or not isinstance(example_id, str) or not example_id.strip():
                    raise ValueError("text and example ID must be non-empty strings")
                if not isinstance(indices, list) or not indices or any(type(index) is not int for index in indices):
                    raise ValueError("labels must be a non-empty integer list")
                if len(indices) != len(set(indices)):
                    raise ValueError("label indices must be unique")
                example_labels = frozenset(LABELS[index] for index in indices)
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise ValueError(f"Invalid GoEmotions row {row_number} in {dataset_path}") from exc
            examples.append(GoEmotionsExample(text=text, labels=example_labels, example_id=example_id))
        if not examples:
            raise ValueError(f"GoEmotions split is empty: {dataset_path}")
        return examples
    labels = load_labels(label_path or dataset_path.with_name("emotions.txt"))
    examples: list[GoEmotionsExample] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            text, raw_labels, example_id = line.split("\t", 2)
            indices = [int(index) for index in raw_labels.split(",")]
            if not text.strip() or not example_id.strip():
                raise ValueError("text and example ID must be non-empty")
            if len(indices) != len(set(indices)):
                raise ValueError("label indices must be unique")
            example_labels = frozenset(labels[index] for index in indices)
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid GoEmotions row {line_number} in {dataset_path}") from exc
        examples.append(GoEmotionsExample(text=text, labels=example_labels, example_id=example_id))
    if not examples:
        raise ValueError(f"GoEmotions split is empty: {dataset_path}")
    return examples


def validate_json_splits(data_dir: Path) -> dict[str, object]:
    """Validate the published Kaggle validation/test JSON pair."""

    paths = {name: Path(data_dir) / f"goemotions_{name}.json" for name in ("val", "test")}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"GoEmotions JSON siblings are incomplete: {missing}")
    splits = {name: load_examples(path) for name, path in paths.items()}
    ids = {name: {row.example_id for row in rows} for name, rows in splits.items()}
    duplicates = {name: len(rows) - len(ids[name]) for name, rows in splits.items()}
    if any(duplicates.values()):
        raise ValueError(f"Duplicate GoEmotions IDs detected within splits: {duplicates}")
    overlap = sorted(ids["val"] & ids["test"])
    if overlap:
        raise ValueError("GoEmotions validation/test ID leakage detected.")
    return {
        "labels": list(LABELS),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "overlap": {"val_test": overlap},
        "duplicates": duplicates,
    }
