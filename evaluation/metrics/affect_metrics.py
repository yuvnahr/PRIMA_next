"""Affect metrics for runtime replay."""

from __future__ import annotations

from collections import Counter
from typing import Any


def emotion_frequency(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count dominant emotion labels."""
    return dict(Counter(str(record.get("emotion", "neutral")) for record in records))


def pad_distributions(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Collect PAD values when records include them."""
    return {
        "valence": [float(record["valence"]) for record in records if "valence" in record],
        "arousal": [float(record["arousal"]) for record in records if "arousal" in record],
        "dominance": [float(record["dominance"]) for record in records if "dominance" in record],
    }


def classification_report(
    expected: list[str],
    predicted: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, precision, recall, F1, and confusion matrix."""
    active_labels = list(labels) if labels is not None else sorted(set(expected) | set(predicted))
    confusion = {actual: {label: 0 for label in active_labels} for actual in active_labels}
    correct = 0
    for actual, guess in zip(expected, predicted, strict=False):
        if actual not in confusion:
            confusion[actual] = {label: 0 for label in active_labels}
        if guess not in confusion[actual]:
            for row in confusion.values():
                row.setdefault(guess, 0)
            active_labels.append(guess)
        confusion[actual][guess] += 1
        if actual == guess:
            correct += 1

    total = len(expected)
    per_label: dict[str, dict[str, float]] = {}
    for label in active_labels:
        true_positive = confusion.get(label, {}).get(label, 0)
        false_positive = sum(row.get(label, 0) for actual, row in confusion.items() if actual != label)
        false_negative = sum(count for guess, count in confusion.get(label, {}).items() if guess != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(confusion.get(label, {}).values()),
        }

    scored_labels = [
        label
        for label, metric in per_label.items()
        if metric["support"] > 0 or sum(row.get(label, 0) for row in confusion.values()) > 0
    ]
    macro_precision = (
        sum(per_label[label]["precision"] for label in scored_labels) / len(scored_labels)
        if scored_labels
        else 0.0
    )
    macro_recall = (
        sum(per_label[label]["recall"] for label in scored_labels) / len(scored_labels)
        if scored_labels
        else 0.0
    )
    macro_f1 = (
        sum(per_label[label]["f1"] for label in scored_labels) / len(scored_labels)
        if scored_labels
        else 0.0
    )
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "per_label": per_label,
        "confusion_matrix": confusion,
        "sample_count": total,
    }
