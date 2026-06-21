"""Types for adaptive memory admission."""

from __future__ import annotations

from dataclasses import dataclass, field

from memory.maintenance.importance_score import ImportanceScore


@dataclass(frozen=True, slots=True)
class ImportanceWeights:
    novelty: float = 0.35
    emotional_salience: float = 0.25
    user_relevance: float = 0.15
    recurrence: float = 0.15
    reflection: float = 0.10

    def normalized(self) -> ImportanceWeights:
        total = self.novelty + self.emotional_salience + self.user_relevance + self.recurrence + self.reflection
        if total <= 0.0:
            return ImportanceWeights()
        return ImportanceWeights(
            novelty=self.novelty / total,
            emotional_salience=self.emotional_salience / total,
            user_relevance=self.user_relevance / total,
            recurrence=self.recurrence / total,
            reflection=self.reflection / total,
        )


@dataclass(frozen=True, slots=True)
class MemoryImportanceConfig:
    threshold: float = 0.55
    top_k: int = 5
    weights: ImportanceWeights = field(default_factory=ImportanceWeights)


@dataclass(frozen=True, slots=True)
class MemoryAdmissionDecision:
    query: str
    score: ImportanceScore
    stored: bool
    threshold: float
    reason: str

    def to_log_record(self) -> dict[str, object]:
        return {
            "query": self.query,
            "novelty": self.score.novelty_score,
            "emotion": self.score.emotional_salience_score,
            "relevance": self.score.user_relevance_score,
            "recurrence": self.score.recurrence_score,
            "reflection": self.score.reflection_score,
            "total_score": self.score.total_score,
            "threshold": round(float(self.threshold), 6),
            "stored": self.stored,
            "reason": self.reason,
        }
