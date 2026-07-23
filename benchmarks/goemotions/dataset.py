"""Load the official GoEmotions TSV splits without copying the dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    """Parse the official headerless ``text, labels, id`` TSV format."""

    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"GoEmotions dataset not found: {dataset_path}")
    labels = load_labels(label_path or dataset_path.with_name("emotions.txt"))
    examples: list[GoEmotionsExample] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            text, raw_labels, example_id = line.split("\t", 2)
            indices = [int(index) for index in raw_labels.split(",")]
            example_labels = frozenset(labels[index] for index in indices)
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid GoEmotions row {line_number} in {dataset_path}") from exc
        examples.append(GoEmotionsExample(text=text, labels=example_labels, example_id=example_id))
    return examples
