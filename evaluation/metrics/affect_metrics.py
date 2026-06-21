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
