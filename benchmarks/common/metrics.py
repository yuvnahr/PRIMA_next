"""Shared metric helpers for benchmark evaluators."""

from __future__ import annotations

from collections.abc import Iterable


def exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 when normalized strings match exactly."""

    return float(prediction.strip().lower() == reference.strip().lower())


def mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty iterable."""

    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)
