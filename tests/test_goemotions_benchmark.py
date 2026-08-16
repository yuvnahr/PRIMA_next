import json
from pathlib import Path

from benchmarks.goemotions.dataset import load_examples, load_labels, validate_json_splits
from benchmarks.goemotions.experiment import _parse_labels
from benchmarks.goemotions.metrics import evaluate, paired_outcomes, probability_metrics
from benchmarks.goemotions.schemas import parse_label_response


def test_goemotions_dataset_and_metrics() -> None:
    fixture = Path(__file__).parent / "fixtures" / "goemotions"
    labels = load_labels(fixture / "emotions.txt")
    examples = load_examples(fixture / "test.tsv", fixture / "emotions.txt")
    assert len(labels) == 28 and labels[-1] == "neutral"
    assert len(examples) == 1
    prediction, error = _parse_labels('{"labels": ["joy", "neutral"]}', labels)
    assert prediction == {"joy", "neutral"} and error is None
    metrics = evaluate([examples[0].labels], [prediction], labels)
    assert metrics["sample_count"] == 1
    assert set(metrics["per_class"]) == set(labels)
    assert metrics["exact_set_accuracy"] == metrics["accuracy"]
    assert "label_cooccurrence" in metrics and "confusion_matrix" not in metrics


def test_strict_response_validation_and_neutral_combinations() -> None:
    labels = load_labels(Path(__file__).parent / "fixtures" / "goemotions" / "emotions.txt")
    for response in (
        "",
        "not json",
        '{"labels":[]}',
        '{"labels":["joy","joy"]}',
        '{"labels":["unknown"]}',
        '{"labels":["joy"],"extra":true}',
    ):
        prediction, error = parse_label_response(response, labels)
        assert not prediction and error
    prediction, error = parse_label_response('{"labels":["neutral","joy"]}', labels)
    assert prediction == {"neutral", "joy"} and error is None


def test_probability_metrics_group_tied_scores() -> None:
    labels = ["joy", "neutral"]
    gold = [frozenset({"joy"}), frozenset({"neutral"})]
    tied = [{"joy": 0.5, "neutral": 0.5}, {"joy": 0.5, "neutral": 0.5}]
    metrics = probability_metrics(gold, tied, labels)
    assert metrics["per_label_pr_auc"] == {"joy": 0.5, "neutral": 0.5}


def test_metrics_remain_unrounded_in_memory() -> None:
    metrics = evaluate([frozenset({"joy"})], [frozenset({"joy", "neutral"})], ["joy", "neutral"])
    assert metrics["micro_f1"] == 2 / 3
    assert metrics["micro_f1"] != round(metrics["micro_f1"], 6)


def test_dataset_rejects_malformed_rows(tmp_path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "goemotions"
    labels_path = tmp_path / "emotions.txt"
    labels_path.write_text((fixture / "emotions.txt").read_text(encoding="utf-8"), encoding="utf-8")
    malformed = tmp_path / "test.tsv"
    malformed.write_text("text\t17,17\tid\n", encoding="utf-8")
    try:
        load_examples(malformed, labels_path)
    except ValueError as exc:
        assert "row 1" in str(exc)
    else:
        raise AssertionError("Duplicate source labels must be rejected.")


def test_kaggle_json_splits_use_canonical_labels(tmp_path) -> None:
    (tmp_path / "goemotions_val.json").write_text(
        json.dumps([{"text": "calm", "labels": [27], "id": "val-1"}]), encoding="utf-8"
    )
    test_path = tmp_path / "goemotions_test.json"
    test_path.write_text(
        json.dumps([{"text": "happy", "labels": [17, 26], "id": "test-1"}]), encoding="utf-8"
    )
    examples = load_examples(test_path)
    assert examples[0].labels == {"joy", "surprise"}
    assert validate_json_splits(tmp_path)["counts"] == {"val": 1, "test": 1}


def test_paired_outcome_groups_separate_parse_recovery() -> None:
    groups = paired_outcomes(
        [frozenset({"joy"}), frozenset({"anger"}), frozenset({"neutral"})],
        [frozenset(), frozenset({"neutral"}), frozenset({"neutral"})],
        [frozenset({"joy"}), frozenset({"anger"}), frozenset({"joy"})],
        [True, False, False],
    )
    assert groups["baseline_validity"] == {"valid_baseline_predictions": 2, "baseline_parse_failures": 1}
    assert groups["affect_outcomes"]["prima_parse_recovery"] == 1
    assert groups["affect_outcomes"]["prima_label_correction"] == 1
    assert groups["affect_outcomes"]["prima_regression"] == 1
