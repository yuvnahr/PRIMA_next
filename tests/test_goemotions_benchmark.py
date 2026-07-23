from benchmarks.goemotions.dataset import DEFAULT_DATASET_PATH, load_examples, load_labels
from benchmarks.goemotions.experiment import _parse_labels
from benchmarks.goemotions.metrics import evaluate


def test_goemotions_dataset_and_metrics() -> None:
    labels = load_labels()
    examples = load_examples()
    assert len(labels) == 28 and labels[-1] == "neutral"
    assert len(examples) == 5427
    prediction, error = _parse_labels('{"labels": ["joy", "neutral"]}', labels)
    assert prediction == {"joy"} and error is None
    metrics = evaluate([examples[0].labels], [prediction], labels)
    assert metrics["sample_count"] == 1
    assert set(metrics["per_class"]) == set(labels)
