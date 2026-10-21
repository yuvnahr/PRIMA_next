"""Official GoEmotions multilabel metrics and deterministic figures."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any

from affect.taxonomies.goemotions import LABELS
from benchmarks.goemotions.schemas import parse_label_response


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
            "prediction_count": tp + fp,
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
    if any(set(row) != set(labels) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
        for value in row.values()
    ) for row in probabilities):
        raise ValueError("Probability rows must contain finite [0, 1] scores for every label.")
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
    axis.set(title="GoEmotions gold/predicted label co-occurrence", xlabel="Predicted label", ylabel="Gold label")
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
    differences = [
        _scores(len(truth & after), len(after - truth), len(truth - after))[2]
        - _scores(len(truth & before), len(before - truth), len(truth - before))[2]
        for truth, before, after in zip(gold, baseline, final, strict=True)
    ]
    deltas = []
    generator = random.Random(seed)  # noqa: S311  # nosec B311
    for _ in range(samples):
        deltas.append(sum(differences[generator.randrange(len(gold))] for _ in gold) / len(gold))
    deltas.sort()
    lower = deltas[int(0.025 * (samples - 1))]
    upper = deltas[int(0.975 * (samples - 1))]
    observed = sum(differences) / len(differences)
    return {"samples": samples, "delta": observed, "lower": lower, "upper": upper}


def probability_eligibility(record: dict[str, Any], dev_hash: str | None = None) -> dict[str, Any]:
    """Label indicators and uncalibrated sigmoid scores cannot establish confidence."""
    unavailable = {"available": False, "confidence": None}
    provenance = record.get("probability_provenance") or {}
    if record.get("probabilities_available") is not True:
        return {**unavailable, "reason": "label_only_or_unavailable_scores"}
    if record.get("parse_error") is not None or record.get("baseline_parse_error") is not None or record.get("execution_failed"):
        return {**unavailable, "reason": "invalid_classification"}
    if not isinstance(provenance, dict) or provenance.get("source") != "trained_probability" or provenance.get("calibrated") is not True:
        return {**unavailable, "reason": "uncalibrated_or_unverified_source"}
    prediction = record.get("prediction") or {}
    if not isinstance(prediction, dict):
        return {**unavailable, "reason": "missing_prediction"}
    for key in ("dev_dataset_sha256", "artifact_sha256"):
        value = provenance.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            return {**unavailable, "reason": "missing_calibration_identity"}
    if (provenance.get("method") != "scalar_temperature" or provenance.get("split") != "dev"
            or not provenance.get("model_id") or provenance.get("model_id") != prediction.get("model_id")
            or not provenance.get("model_revision") or provenance.get("model_revision") != prediction.get("model_revision")
            or dev_hash is None or provenance["dev_dataset_sha256"] != dev_hash):
        return {**unavailable, "reason": "calibration_model_or_dev_identity_mismatch"}
    row = prediction.get("probabilities")
    if not isinstance(row, dict) or set(row) != set(LABELS) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
        for value in row.values()
    ):
        return {**unavailable, "reason": "incomplete_or_invalid_28_label_scores"}
    return {"available": True, "confidence": max(row.values()), "reason": "dev_calibrated_full_label_scores", "provenance": provenance}


def correction_metrics(gold: list[frozenset[str]], baseline: list[frozenset[str]], final: list[frozenset[str]]) -> dict[str, Any]:
    if not len(gold) == len(baseline) == len(final):
        raise ValueError("Correction inputs must have matching lengths.")
    deltas, added, correct_added = [], 0, 0
    for truth, before, after in zip(gold, baseline, final, strict=True):
        deltas.append(_scores(len(truth & after), len(after - truth), len(truth - after))[2]
                      - _scores(len(truth & before), len(before - truth), len(truth - before))[2])
        added += len(after - before)
        correct_added += len((after - before) & truth)
    return {
        "available": bool(gold), "sample_count": len(gold), "partial_f1_gains": sum(d > 0 for d in deltas),
        "partial_f1_harms": sum(d < 0 for d in deltas), "partial_f1_unchanged": sum(d == 0 for d in deltas),
        "mean_sample_f1_delta": _mean(deltas) if deltas else None, "added_labels": added, "correct_added_labels": correct_added,
        "added_label_precision": correct_added / added if added else None,
        "improved_after_label_correction": sum(b != g and a == g for g, b, a in zip(gold, baseline, final, strict=True)),
        "regressed_after_label_correction": sum(b == g and a != g for g, b, a in zip(gold, baseline, final, strict=True)),
        "scope": "valid_paired_same_response_labels_not_runtime_reflection",
    }


def same_response_eligible(record: dict[str, Any]) -> bool:
    if "baseline_labels" not in record or record.get("baseline_parse_error") is not None or record.get("parse_error") is not None or record.get("execution_failed"):
        return False
    if record.get("same_response_comparison_eligible") is False:
        return False
    raw = record.get("raw_model_response", record.get("raw_response"))
    if not isinstance(raw, str):
        return False
    baseline, error = parse_label_response(raw)
    expected_hash = record.get("raw_model_response_hash", record.get("response_hash"))
    return error is None and baseline == frozenset(record["baseline_labels"]) and (
        expected_hash is None or hashlib.sha256(raw.encode()).hexdigest() == expected_hash
    )


def performance_metrics(records: list[dict[str, Any]], timing: dict[str, Any] | None = None) -> dict[str, Any]:
    latencies = [float(row.get("latency_ms") or 0) for row in records]
    if any(not math.isfinite(value) or value < 0 for value in latencies):
        raise ValueError("Observed case latency must be finite and nonnegative")
    latency_complete = bool(records) and all(row.get("latency_ms") is not None for row in records)
    summed = sum(latencies) / 1000
    logical_counts = []
    for record in records:
        usage = record.get("provider_usage") or {}
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        if total is None and isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int) and isinstance(usage.get("completion_tokens"), int):
            total = usage["prompt_tokens"] + usage["completion_tokens"]
        logical_counts.append(total if isinstance(total, int) and not isinstance(total, bool) and total >= 0 else None)
    logical_total = sum(value or 0 for value in logical_counts)
    logical_complete = bool(records) and all(value is not None for value in logical_counts)
    observed = [row.get("classification_provider") or {} for row in records]
    available = bool(records) and all(row.get("available") is True for row in observed)
    windows: dict[str, list[float]] = {}
    for record, row in zip(records, observed, strict=True):
        if row.get("inference_started") is not None and row.get("inference_finished") is not None:
            windows.setdefault(str(record.get("attempt_id", "current")), []).extend([row["inference_started"], row["inference_finished"]])
    inference_wall = sum(max(values) - min(values) for values in windows.values()) if windows else None
    successes = sum(row.get("successes", 0) for row in observed) if available else None
    actual_tokens = sum(row.get("actual_tokens", 0) for row in observed) if available and all(row.get("actual_tokens_available") for row in observed) else None
    active = timing.get("active_wall_seconds") if timing and timing.get("complete") else None
    return {
        "total_latency_ms": sum(latencies), "mean_latency_ms": sum(latencies) / len(records) if records else 0.0,
        "examples_per_summed_latency_second": len(records) / summed if summed and latency_complete else None,
        "summed_latency_complete": latency_complete,
        "wall_throughput_examples_per_second": len(records) / active if active else None,
        "active_wall_seconds": active, "logical_tokens": logical_total if logical_complete else None,
        "logical_tokens_available": logical_complete,
        "logical_tokens_per_summed_latency_second": logical_total / summed if summed and logical_complete else None,
        "provider_measurements_available": available,
        "classification_provider_calls": sum(row.get("attempts", 0) for row in observed) if available else None,
        "logical_classification_provider_calls": sum(row["logical_classification_provider_calls"] for row in records) if all("logical_classification_provider_calls" in row for row in records) else None,
        "successful_uncached_requests": successes,
        "response_cache_hits": sum(row.get("cache_hits", 0) for row in observed) if available else None,
        "provider_retries": sum(row.get("retries", 0) for row in observed) if available else None,
        "actual_uncached_tokens": actual_tokens, "inference_window_seconds": inference_wall,
        "provider_requests_per_inference_second": successes / inference_wall if successes is not None and inference_wall else None,
        "actual_tokens_per_inference_second": actual_tokens / inference_wall if actual_tokens is not None and inference_wall else None,
        "measurement": "wall counts all completed checkpoints, including failures; logical tokens include cache hits; provider windows grouped by active attempt",
        "throughput_examples_per_second": len(records) / summed if summed else 0.0,
        "observed_logical_tokens_sum": logical_total,
        "observed_logical_tokens_per_summed_latency_second": logical_total / summed if summed else 0.0,
        "total_tokens": logical_total, "tokens_per_second": logical_total / summed if summed else 0.0,
        "deprecated_aliases": {"throughput_examples_per_second": "legacy_examples_over_observed_summed_latency", "total_tokens": "observed_logical_tokens_sum", "tokens_per_second": "observed_logical_tokens_per_summed_latency_second"},
    }
