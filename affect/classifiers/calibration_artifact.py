"""Identity contract for GoEmotions encoder calibration and thresholds."""

from __future__ import annotations

import math
import re
from typing import Any

from affect.taxonomies.goemotions import LABELS
from config.frozen_json import fingerprint


def encoder_identity(
    model_id: str, revision: str, max_length: int, dev_hash: str, dev_ids_hash: str,
) -> dict[str, Any]:
    if not model_id or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Calibration requires the full immutable model commit hash")
    if any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in (dev_hash, dev_ids_hash)):
        raise ValueError("Calibration requires development data and ordered ID hashes")
    return {
        "model_id": model_id, "model_revision": revision, "split": "dev",
        "taxonomy": "goemotions", "ordered_labels_sha256": fingerprint(list(LABELS)),
        "preprocessing": {"version": "raw-tokenizer-truncate-v1", "max_length": max_length},
        "dev_dataset_sha256": dev_hash, "dev_ids_sha256": dev_ids_hash,
    }


def finite_parameter(value: Any, *, lower: float, upper: float) -> float:
    if type(value) not in (int, float) or not lower <= value <= upper or not math.isfinite(value):
        raise ValueError("Invalid calibration parameter")
    return float(value)
