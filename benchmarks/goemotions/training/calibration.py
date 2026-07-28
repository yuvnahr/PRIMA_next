"""Development-only scalar temperature scaling serialization."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks.goemotions.training.data import load_split, multi_hot, validate_splits


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


def write_calibration(path: Path, result: dict[str, Any], *, model_id: str, dev_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**result, "model_id": model_id, "dev_dataset_sha256": dev_hash}, indent=2, sort_keys=True), encoding="utf-8")


def fit_calibration_from_dev(data_dir: Path, logits_path: Path, output_path: Path, *, model_id: str) -> dict[str, Any]:
    """Fit scalar temperature from development logits in official development order."""
    labels = validate_splits(data_dir)["labels"]
    logits = json.loads(logits_path.read_text(encoding="utf-8"))
    dev = load_split(data_dir, "dev")
    if not isinstance(logits, list) or len(logits) != len(dev):
        raise ValueError("Development logits must match the official dev split length.")
    result = fit_temperature(logits, [multi_hot(row, labels) for row in dev])
    write_calibration(output_path, result, model_id=model_id, dev_hash=validate_splits(data_dir)["hashes"]["dev"])
    return result
