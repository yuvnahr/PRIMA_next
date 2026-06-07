"""Configurable hybrid score fusion."""

from __future__ import annotations

from dataclasses import dataclass, field

from memory.retrieval.retrieval_result import RetrievalResult


@dataclass(frozen=True, slots=True)
class HybridFusionConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "dense": 0.4,
            "sparse": 0.3,
            "temporal": 0.15,
            "graph": 0.15,
        }
    )


class HybridFusion:
    def __init__(self, config: HybridFusionConfig | None = None) -> None:
        self.config = config or HybridFusionConfig()

    def fuse(self, results_by_strategy: dict[str, list[RetrievalResult]], top_k: int = 5) -> list[RetrievalResult]:
        merged: dict[str, RetrievalResult] = {}
        scores: dict[str, dict[str, float]] = {}

        for strategy_name, results in results_by_strategy.items():
            for result in results:
                merged[result.note.id] = result
                scores.setdefault(result.note.id, {})[strategy_name] = max(
                    scores.setdefault(result.note.id, {}).get(strategy_name, 0.0),
                    result.score,
                )

        fused: list[RetrievalResult] = []
        total_weight = sum(self.config.weights.values()) or 1.0
        for note_id, result in merged.items():
            strategy_scores = scores[note_id]
            weighted = sum(
                strategy_scores.get(name, 0.0) * weight
                for name, weight in self.config.weights.items()
            ) / total_weight
            fused.append(result.with_score(weighted, strategy_scores=strategy_scores))

        return sorted(fused, key=lambda item: item.score, reverse=True)[:top_k]
