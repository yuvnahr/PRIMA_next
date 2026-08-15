"""Configurable hybrid score fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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

    @classmethod
    def from_file(cls, path: str | Path = "config/retrieval.yaml") -> HybridFusionConfig:
        """Load simple retrieval weights from YAML-like key/value config."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        values: dict[str, float] = {}
        for raw_line in config_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                values[key.strip()] = float(value.strip())
            except ValueError:
                continue
        weights = {
            "dense": values.get("dense_weight", cls().weights["dense"]),
            "sparse": values.get("sparse_weight", cls().weights["sparse"]),
            "temporal": values.get("temporal_weight", cls().weights["temporal"]),
            "graph": values.get("graph_weight", cls().weights["graph"]),
        }
        return cls(weights=weights)


class HybridFusion:
    def __init__(self, config: HybridFusionConfig | None = None) -> None:
        self.config = config or HybridFusionConfig()

    def fuse(self, results_by_strategy: dict[str, list[RetrievalResult]], top_k: int = 5) -> list[RetrievalResult]:
        merged: dict[str, RetrievalResult] = {}
        scores: dict[str, dict[str, float]] = {}
        normalized_by_strategy = self._normalize(results_by_strategy)

        for strategy_name, results in normalized_by_strategy.items():
            for result in results:
                merged[result.note.id] = result
                scores.setdefault(result.note.id, {})[strategy_name] = max(
                    scores.setdefault(result.note.id, {}).get(strategy_name, 0.0),
                    result.score,
                )

        fused: list[RetrievalResult] = []
        active_names = set(results_by_strategy)
        total_weight = sum(
            weight for name, weight in self.config.weights.items() if name in active_names
        ) or 1.0
        for note_id, result in merged.items():
            strategy_scores = scores[note_id]
            weighted = sum(
                strategy_scores.get(name, 0.0) * weight
                for name, weight in self.config.weights.items()
            ) / total_weight
            fused.append(result.with_score(weighted, strategy_scores=strategy_scores))

        return sorted(fused, key=lambda item: item.score, reverse=True)[:top_k]

    def _normalize(self, results_by_strategy: dict[str, list[RetrievalResult]]) -> dict[str, list[RetrievalResult]]:
        normalized: dict[str, list[RetrievalResult]] = {}
        for strategy_name, results in results_by_strategy.items():
            if not results:
                normalized[strategy_name] = []
                continue
            raw_scores = [result.score for result in results]
            minimum = min(raw_scores)
            maximum = max(raw_scores)
            if maximum <= minimum:
                normalized[strategy_name] = results
                continue
            normalized[strategy_name] = [
                result.with_score((result.score - minimum) / (maximum - minimum), {strategy_name: result.score})
                for result in results
            ]
        return normalized
