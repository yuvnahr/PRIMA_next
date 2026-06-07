"""Evolution policy configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvolutionPolicy:
    min_cluster_size: int = 2
    max_summary_words: int = 40
    min_salience_for_evolution: float = 0.0
