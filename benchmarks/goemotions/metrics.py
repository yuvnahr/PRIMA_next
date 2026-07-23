"""Official GoEmotions multilabel metrics and deterministic figures."""

from __future__ import annotations

from collections import Counter
from typing import Any


def evaluate(gold: list[frozenset[str]], predicted: list[frozenset[str]], labels: list[str]) -> dict[str, Any]:
    """Match Google's binary multilabel precision/recall/F1 aggregation."""

    if len(gold) != len(predicted):
        raise ValueError("Gold and prediction counts must match.")
    per_class: dict[str, dict[str, float | int]] = {}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for label in labels:
        tp = sum(label in actual and label in guess for actual, guess in zip(gold, predicted, strict=True))
        fp = sum(label not in actual and label in guess for actual, guess in zip(gold, predicted, strict=True))
        fn = sum(label in actual and label not in guess for actual, guess in zip(gold, predicted, strict=True))
        tn = len(gold) - tp - fp - fn
        precision, recall, f1 = _scores(tp, fp, fn)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "accuracy": _ratio(tp + tn, len(gold)), "support": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
    micro_precision, micro_recall, micro_f1 = _scores(**totals)
    supports = [int(per_class[label]["support"]) for label in labels]
    exact_accuracy = _ratio(sum(actual == guess for actual, guess in zip(gold, predicted, strict=True)), len(gold))
    return {
        "sample_count": len(gold),
        "accuracy": exact_accuracy,
        "macro_precision": _mean(float(per_class[label]["precision"]) for label in labels),
        "macro_recall": _mean(float(per_class[label]["recall"]) for label in labels),
        "macro_f1": _mean(float(per_class[label]["f1"]) for label in labels),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "weighted_precision": _weighted(per_class, labels, supports, "precision"),
        "weighted_recall": _weighted(per_class, labels, supports, "recall"),
        "weighted_f1": _weighted(per_class, labels, supports, "f1"),
        "per_class": per_class,
        "label_distribution": dict(Counter(label for row in gold for label in row)),
        "prediction_distribution": dict(Counter(label for row in predicted for label in row)),
        "confusion_matrix": _cooccurrence(gold, predicted, labels),
    }


def render_figures(metrics: dict[str, Any], labels: list[str], confusion_path: str, label_path: str, prediction_path: str) -> None:
    """Render the requested diagnostics with matplotlib, already used by GoEmotions."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = [[metrics["confusion_matrix"][actual][guess] for guess in labels] for actual in labels]
    fig, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(title="GoEmotions label co-occurrence", xlabel="Predicted label", ylabel="Gold label")
    axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    axis.set_yticks(range(len(labels)), labels, fontsize=6)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)
    for path, title, distribution in ((label_path, "Gold label distribution", metrics["label_distribution"]), (prediction_path, "Prediction distribution", metrics["prediction_distribution"])):
        fig, axis = plt.subplots(figsize=(12, 4))
        axis.bar(labels, [distribution.get(label, 0) for label in labels])
        axis.set(title=title, ylabel="Count")
        axis.tick_params(axis="x", rotation=90, labelsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)


def _scores(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(precision, 6), round(recall, 6), round(f1, 6)


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _mean(values: Any) -> float:
    values = list(values)
    return _ratio(sum(values), len(values))


def _weighted(per_class: dict[str, dict[str, float | int]], labels: list[str], supports: list[int], metric: str) -> float:
    total = sum(supports)
    return _ratio(sum(float(per_class[label][metric]) * support for label, support in zip(labels, supports, strict=True)), total)


def _cooccurrence(gold: list[frozenset[str]], predicted: list[frozenset[str]], labels: list[str]) -> dict[str, dict[str, int]]:
    return {actual: {guess: sum(actual in truth and guess in prediction for truth, prediction in zip(gold, predicted, strict=True)) for guess in labels} for actual in labels}
