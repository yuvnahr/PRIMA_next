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
    errors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the runtime result into JSON-friendly values."""
        return {
            "final_response": self.final_response,
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
            "confidence_score": self.confidence_score,
            "latency_ms": self.latency_ms,
            "errors": list(self.errors),
        }
