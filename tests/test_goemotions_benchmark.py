from pathlib import Path

from benchmarks.goemotions.dataset import load_examples, load_labels
from benchmarks.goemotions.experiment import _parse_labels
from benchmarks.goemotions.metrics import evaluate


def test_goemotions_dataset_and_metrics() -> None:
    fixture = Path(__file__).parent / "fixtures" / "goemotions"
    labels = load_labels(fixture / "emotions.txt")
    examples = load_examples(fixture / "test.tsv", fixture / "emotions.txt")
    assert len(labels) == 28 and labels[-1] == "neutral"
    assert len(examples) == 1
    prediction, error = _parse_labels('{"labels": ["joy", "neutral"]}', labels)
    assert prediction == set() and error == "neutral cannot coexist with non-neutral labels"
    metrics = evaluate([examples[0].labels], [prediction], labels)
    assert metrics["sample_count"] == 1
    assert set(metrics["per_class"]) == set(labels)
