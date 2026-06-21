"""Runtime metrics model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    """Serializable metrics captured for one runtime turn."""

    latency_ms: float
    retrieval_count: int
    memory_creation_count: int
    reflection_trigger_count: int
    emotion_classification: str
    confidence_score: float
    planning_success: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to plain values."""
        return {
            "latency_ms": self.latency_ms,
            "retrieval_count": self.retrieval_count,
            "memory_creation_count": self.memory_creation_count,
            "reflection_trigger_count": self.reflection_trigger_count,
            "emotion_classification": self.emotion_classification,
            "confidence_score": self.confidence_score,
            "planning_success": self.planning_success,
        }
