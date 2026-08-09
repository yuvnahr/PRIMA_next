"""Official GoEmotions multilabel metrics and deterministic figures."""

from __future__ import annotations

from collections import Counter
from typing import Any

from affect.taxonomies.goemotions import LABELS


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
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": _ratio(tp + tn, len(gold)),
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
    micro_precision, micro_recall, micro_f1 = _scores(**totals)
    supports = [int(per_class[label]["support"]) for label in labels]
    exact_accuracy = _ratio(sum(actual == guess for actual, guess in zip(gold, predicted, strict=True)), len(gold))
    sample_scores = [
        _scores(len(actual & guess), len(guess - actual), len(actual - guess))
        for actual, guess in zip(gold, predicted, strict=True)
    ]
    cooccurrence = _cooccurrence(gold, predicted, labels)
    total_labels = max(1, len(gold) * len(labels))
    return {
        "sample_count": len(gold),
        "exact_set_accuracy": exact_accuracy,
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
        "label_cooccurrence": cooccurrence,
        "row_normalized_cooccurrence": {
            label: {
                guess: _ratio(value, sum(cooccurrence[label].values())) for guess, value in cooccurrence[label].items()
            }
            for label in labels
        },
        "hamming_loss": _ratio(
            sum(len(actual ^ guess) for actual, guess in zip(gold, predicted, strict=True)), total_labels
        ),
        "sample_precision": _mean(score[0] for score in sample_scores),
        "sample_recall": _mean(score[1] for score in sample_scores),
        "sample_f1": _mean(score[2] for score in sample_scores),
        "jaccard": _mean(
            _ratio(len(actual & guess), len(actual | guess)) for actual, guess in zip(gold, predicted, strict=True)
        ),
        "label_cardinality": _mean(len(row) for row in gold),
        "prediction_cardinality": _mean(len(row) for row in predicted),
        "cardinality_error": _mean(
            abs(len(actual) - len(guess)) for actual, guess in zip(gold, predicted, strict=True)
        ),
        "empty_prediction_count": sum(not row for row in predicted),
        "neutral_only_rate": _ratio(sum(row == {"neutral"} for row in predicted), len(predicted)),
        "multi_label_prediction_rate": _ratio(sum(len(row) > 1 for row in predicted), len(predicted)),
    }


def probability_metrics(
    gold: list[frozenset[str]], probabilities: list[dict[str, float]], labels: list[str]
) -> dict[str, Any]:
    """Probability-only diagnostics; callers must not use this for label-only systems."""
    if len(gold) != len(probabilities):
        raise ValueError("Gold and probability counts must match.")
    values = [
        (label in truth, float(row[label])) for truth, row in zip(gold, probabilities, strict=True) for label in labels
    ]
    brier = _mean((score - int(target)) ** 2 for target, score in values)
    per_label = {
        label: _pr_auc([(label in truth, float(row[label])) for truth, row in zip(gold, probabilities, strict=True)])
        for label in labels
    }
    valid = [value for value in per_label.values() if value is not None]
    return {
        "available": True,
        "brier_score": brier,
        "expected_calibration_error": _ece(values),
        "per_label_pr_auc": per_label,
        "macro_pr_auc": _mean(valid) if valid else None,
        "micro_pr_auc": _pr_auc(values),
    }


def render_figures(
    metrics: dict[str, Any], labels: list[str], confusion_path: str, label_path: str, prediction_path: str
) -> None:
    """Render the requested diagnostics with matplotlib, already used by GoEmotions."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = [[metrics["label_cooccurrence"][actual][guess] for guess in labels] for actual in labels]
    fig, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(title="GoEmotions label co-occurrence", xlabel="Predicted label", ylabel="Gold label")
    axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    axis.set_yticks(range(len(labels)), labels, fontsize=6)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)
    for path, title, distribution in (
        (label_path, "Gold label distribution", metrics["label_distribution"]),
        (prediction_path, "Prediction distribution", metrics["prediction_distribution"]),
    ):
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
    return precision, recall, f1


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Any) -> float:
    values = list(values)
    return _ratio(sum(values), len(values))


def _weighted(
    per_class: dict[str, dict[str, float | int]], labels: list[str], supports: list[int], metric: str
) -> float:
    total = sum(supports)
    return _ratio(
        sum(float(per_class[label][metric]) * support for label, support in zip(labels, supports, strict=True)), total
    )


def _cooccurrence(
    gold: list[frozenset[str]], predicted: list[frozenset[str]], labels: list[str]
) -> dict[str, dict[str, int]]:
    return {
        actual: {
            guess: sum(
                actual in truth and guess in prediction for truth, prediction in zip(gold, predicted, strict=True)
            )
            for guess in labels
        }
        for actual in labels
    }


def _pr_auc(values: list[tuple[bool, float]]) -> float | None:
    positives = sum(target for target, _ in values)
    if not positives:
        return None
    tp = fp = 0
    previous_recall = area = 0.0
    ranked = sorted(values, key=lambda item: item[1], reverse=True)
    index = 0
    while index < len(ranked):
        score = ranked[index][1]
        tied = []
        while index < len(ranked) and ranked[index][1] == score:
            tied.append(ranked[index][0])
            index += 1
        tp += sum(tied)
        fp += len(tied) - sum(tied)
        recall, precision = tp / positives, tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def _ece(values: list[tuple[bool, float]], bins: int = 10) -> float:
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        bucket = [
            (target, score)
            for target, score in values
            if lower <= score < upper or (index == bins - 1 and score == upper)
        ]
        if bucket:
            error += (
                len(bucket)
                / len(values)
                * abs(_mean(score for _, score in bucket) - _mean(int(target) for target, _ in bucket))
            )
    return error


def paired_outcomes(
    gold: list[frozenset[str]],
    baseline: list[frozenset[str]],
    final: list[frozenset[str]],
    baseline_parse_errors: list[bool],
) -> dict[str, dict[str, int]]:
    """Partition paired affect-layer results into mutually exclusive groups."""

    if not (len(gold) == len(baseline) == len(final) == len(baseline_parse_errors)):
        raise ValueError("Paired outcome inputs must have equal lengths.")
    validity = {"valid_baseline_predictions": 0, "baseline_parse_failures": 0}
    changes = {
        "prima_parse_recovery": 0,
        "unrecovered_parse_failure": 0,
        "prima_label_correction": 0,
        "prima_regression": 0,
        "changed_other": 0,
        "unchanged_prediction": 0,
    }
    for truth, before, after, parse_failed in zip(gold, baseline, final, baseline_parse_errors, strict=True):
        validity["baseline_parse_failures" if parse_failed else "valid_baseline_predictions"] += 1
        if parse_failed and after:
            changes["prima_parse_recovery"] += 1
        elif parse_failed:
            changes["unrecovered_parse_failure"] += 1
        elif before == after:
            changes["unchanged_prediction"] += 1
        elif before != truth and after == truth:
            changes["prima_label_correction"] += 1
        elif before == truth and after != truth:
            changes["prima_regression"] += 1
        else:
            changes["changed_other"] += 1
    if sum(validity.values()) != len(gold) or sum(changes.values()) != len(gold):
        raise RuntimeError("Paired GoEmotions outcome groups do not reconcile.")
    return {"baseline_validity": validity, "affect_outcomes": changes}


def paired_bootstrap_sample_f1(
    gold: list[frozenset[str]],
    baseline: list[frozenset[str]],
    final: list[frozenset[str]],
    *,
    seed: int = 13,
    samples: int = 1000,
) -> dict[str, float | int]:
    """Percentile CI for same-response sample-F1 change."""

    import random

    if not (len(gold) == len(baseline) == len(final)):
        raise ValueError("Paired bootstrap inputs must have equal lengths.")
    if not gold or samples < 1:
        return {"samples": samples, "delta": 0.0, "lower": 0.0, "upper": 0.0}
    deltas = []
    generator = random.Random(seed)  # noqa: S311  # nosec B311
    for _ in range(samples):
        indices = [generator.randrange(len(gold)) for _ in gold]
        before = evaluate([gold[i] for i in indices], [baseline[i] for i in indices], list(LABELS))["sample_f1"]
        after = evaluate([gold[i] for i in indices], [final[i] for i in indices], list(LABELS))["sample_f1"]
        deltas.append(after - before)
    deltas.sort()
    lower = deltas[int(0.025 * (samples - 1))]
    upper = deltas[int(0.975 * (samples - 1))]
    observed = evaluate(gold, final, list(LABELS))["sample_f1"] - evaluate(gold, baseline, list(LABELS))["sample_f1"]
    return {"samples": samples, "delta": observed, "lower": lower, "upper": upper}
