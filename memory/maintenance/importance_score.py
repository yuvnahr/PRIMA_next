"""Normalized importance score for memory admission."""

from __future__ import annotations

from dataclasses import dataclass


def clamp_score(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 6)


@dataclass(frozen=True, slots=True)
class ImportanceScore:
    novelty_score: float
    emotional_salience_score: float
    user_relevance_score: float
    recurrence_score: float
    reflection_score: float
    total_score: float

    def __post_init__(self) -> None:
        for field_name in (
            "novelty_score",
            "emotional_salience_score",
            "user_relevance_score",
            "recurrence_score",
            "reflection_score",
            "total_score",
        ):
            object.__setattr__(self, field_name, clamp_score(getattr(self, field_name)))

    def to_dict(self) -> dict[str, float]:
        return {
            "novelty_score": self.novelty_score,
            "emotional_salience_score": self.emotional_salience_score,
            "user_relevance_score": self.user_relevance_score,
            "recurrence_score": self.recurrence_score,
            "reflection_score": self.reflection_score,
            "total_score": self.total_score,
        }
