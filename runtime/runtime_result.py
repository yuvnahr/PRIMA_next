"""Runtime result model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    """Public result returned by PrimaRuntime.process."""

    final_response: str
    affect_state: dict[str, Any]
    retrieved_memories: tuple[Any, ...]
    memory_notes_created: tuple[Any, ...]
    reflection_triggered: bool
    confidence_score: float
    latency_ms: float
    prediction_before_reflection: str = ""
    prediction_after_reflection: str = ""
    errors: tuple[str, ...] = field(default_factory=tuple)
    reflection_reasons: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    reflection_before_confidence: float = 0.0
    reflection_after_confidence: float = 0.0
    reflection_utility_score: float = 0.0
    correction_count: int = 0
    memory_admission: dict[str, Any] = field(default_factory=dict)
    answer_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the runtime result into JSON-friendly values."""
        return {
            "final_response": self.final_response,
            "prediction_before_reflection": self.prediction_before_reflection,
            "prediction_after_reflection": self.prediction_after_reflection,
            "affect_state": dict(self.affect_state),
            "retrieved_memories": [
                item.note.to_metadata() if hasattr(item, "note") and hasattr(item.note, "to_metadata") else str(item)
                for item in self.retrieved_memories
            ],
            "memory_notes_created": [
                note.to_metadata() if hasattr(note, "to_metadata") else str(note)
                for note in self.memory_notes_created
            ],
            "reflection_triggered": self.reflection_triggered,
            "reflection_reasons": [dict(reason) for reason in self.reflection_reasons],
            "reflection_before_confidence": self.reflection_before_confidence,
            "reflection_after_confidence": self.reflection_after_confidence,
            "reflection_utility_score": self.reflection_utility_score,
            "correction_count": self.correction_count,
            "memory_admission": dict(self.memory_admission),
            "answer_diagnostics": dict(self.answer_diagnostics),
            "confidence_score": self.confidence_score,
            "latency_ms": self.latency_ms,
            "errors": list(self.errors),
        }
