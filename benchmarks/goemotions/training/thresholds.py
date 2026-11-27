"""Development-only deterministic threshold selection."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from affect.classifiers.calibration_artifact import finite_parameter
from affect.taxonomies.goemotions import LABELS
from benchmarks.goemotions.training.data import load_encoder_dev_predictions
from config.frozen_json import freeze_json


def select_thresholds(probabilities: Sequence[Mapping[str, float]], gold: Sequence[frozenset[str]], labels: Sequence[str], grid: Sequence[float] = tuple(i / 100 for i in range(10, 91, 5)), min_support: int = 5) -> dict[str, Any]:
    if len(probabilities) != len(gold):
        raise ValueError("Development probabilities and labels must have equal length.")
    result: dict[str, Any] = {"version": 1, "objective": "per_label_f1", "thresholds": {}, "support": {}}
    for label in labels:
        support = sum(label in row for row in gold)
        result["support"][label] = support
        if support < min_support:
            result["thresholds"][label] = 0.5
            continue
        candidates = [(threshold, _f1(sum(label in truth and row[label] >= threshold for row, truth in zip(probabilities, gold, strict=True)), sum(label not in truth and row[label] >= threshold for row, truth in zip(probabilities, gold, strict=True)), sum(label in truth and row[label] < threshold for row, truth in zip(probabilities, gold, strict=True)))) for threshold in grid]
        result["thresholds"][label] = max(candidates, key=lambda item: (item[1], -item[0]))[0]
    return result


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def write_thresholds(path: Path, result: Mapping[str, Any], *, model_id: str, dev_hash: str) -> None:
    """Persist a versioned artifact produced solely from development rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**result, "model_id": model_id, "dev_dataset_sha256": dev_hash}, indent=2, sort_keys=True), encoding="utf-8")


def fit_thresholds_from_dev(
    data_dir: Path, probabilities_path: Path, output_path: Path, *, model_id: str, model_revision: str,
    min_support: int = 5, max_length: int = 128, temperature_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Fit encoder thresholds against frozen development probabilities."""
    transform = {"method": "sigmoid", "temperature_fingerprint": temperature_fingerprint}
    identity, dev, predictions = load_encoder_dev_predictions(
        data_dir, probabilities_path, model_id=model_id, revision=model_revision,
        max_length=max_length, representation="probabilities", probability_transform=transform,
    )
    probabilities = predictions["rows"]
    for row in probabilities:
        if not isinstance(row, dict) or set(row) != set(LABELS):
            raise ValueError("Development probabilities must cover every official label")
        for value in row.values():
            finite_parameter(value, lower=0.0, upper=1.0)
    result = select_thresholds(probabilities, [row.labels for row in dev], LABELS, min_support=min_support)
    return freeze_json(output_path, {**result, **identity, "version": 2, "method": "per_label_f1",
                                    "probability_transform": transform, "input_fingerprint": predictions["fingerprint"]})
