"""Emotional salience scoring."""

from __future__ import annotations


def compute_salience(intensity: float, novelty: float, volatility: float) -> float:
    """Blend intensity, novelty, and volatility into a bounded salience score."""
    score = intensity * 0.55 + novelty * 0.25 + volatility * 0.20
    return round(max(0.0, min(1.0, score)), 6)
