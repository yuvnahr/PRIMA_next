"""Retrieval result model."""

from __future__ import annotations

from dataclasses import dataclass, field

from memory.memory_note import MemoryNote


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    note: MemoryNote
    score: float
    strategy_scores: dict[str, float] = field(default_factory=dict)
    explanation: dict[str, object] = field(default_factory=dict)

    def with_score(self, score: float, strategy_scores: dict[str, float] | None = None) -> "RetrievalResult":
        return RetrievalResult(
            note=self.note,
            score=max(0.0, min(1.0, score)),
            strategy_scores=strategy_scores or dict(self.strategy_scores),
            explanation=dict(self.explanation),
        )
