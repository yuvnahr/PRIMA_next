"""Development-only scalar temperature scaling serialization."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from affect.classifiers.calibration_artifact import finite_parameter
from affect.taxonomies.goemotions import LABELS
from benchmarks.goemotions.training.data import load_encoder_dev_predictions, multi_hot
from config.frozen_json import freeze_json


def fit_temperature(logits: Sequence[Sequence[float]], targets: Sequence[Sequence[float]], candidates: Sequence[float] = tuple(i / 20 for i in range(10, 81))) -> dict[str, Any]:
    if len(logits) != len(targets) or not logits:
        raise ValueError("Non-empty matching development logits and targets are required.")
    scores = [(temperature, _nll(logits, targets, temperature)) for temperature in candidates if temperature > 0]
    temperature, loss = min(scores, key=lambda item: item[1])
    return {"version": 1, "method": "scalar_temperature", "temperature": temperature, "dev_nll": loss}


def _nll(logits: Sequence[Sequence[float]], targets: Sequence[Sequence[float]], temperature: float) -> float:
    terms = []
    for row, target in zip(logits, targets, strict=True):
        for logit, value in zip(row, target, strict=True):
            probability = 1 / (1 + math.exp(-max(-60.0, min(60.0, logit / temperature))))
            terms.append(-(value * math.log(probability + 1e-12) + (1 - value) * math.log(1 - probability + 1e-12)))
    return sum(terms) / len(terms)


def write_calibration(path: Path, result: dict[str, Any], *, identity: dict[str, Any]) -> dict[str, Any]:
    temperature = finite_parameter(result.get("temperature"), lower=0.0, upper=float("inf"))
    if temperature <= 0:
        raise ValueError("Calibration temperature must be positive")
    return freeze_json(path, {**result, **identity, "version": 2, "method": "scalar_temperature"})


def fit_calibration_from_dev(
    data_dir: Path, logits_path: Path, output_path: Path, *, model_id: str, model_revision: str, max_length: int = 128,
) -> dict[str, Any]:
    """Fit scalar temperature from a frozen dev logit artifact."""
    identity, dev, predictions = load_encoder_dev_predictions(
        data_dir, logits_path, model_id=model_id, revision=model_revision,
        max_length=max_length, representation="logits",
    )
    logits = predictions["rows"]
    if any(not isinstance(row, list) or len(row) != len(LABELS) for row in logits):
        raise ValueError("Development logits must cover every ordered label")
    for row in logits:
        for value in row:
            finite_parameter(value, lower=float("-inf"), upper=float("inf"))
    result = fit_temperature(logits, [multi_hot(row, list(LABELS)) for row in dev])
    return write_calibration(output_path, {**result, "input_fingerprint": predictions["fingerprint"]}, identity=identity)
