"""Runtime context for one PRIMA-NEXT processing cycle."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from state.cognitive_state import CognitiveState
from state.emotional_state import EmotionalState


@dataclass(slots=True)
class RuntimeContext:
    """Shared runtime state derived from workflow-owned execution context."""

    session_id: str = field(default_factory=lambda: f"session_{uuid.uuid4()}")
    turn_id: str = field(default_factory=lambda: f"turn_{uuid.uuid4()}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cognitive_state: CognitiveState = field(default_factory=CognitiveState)
    emotional_state: EmotionalState = field(default_factory=EmotionalState)
    retrieved_memories: tuple[Any, ...] = ()
    active_plan: Any | None = None
    reflection_signals: tuple[Any, ...] = ()
    confidence_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the runtime context into JSON-friendly values."""
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp.isoformat(),
            "cognitive_state": {
                "goal_state": dict(self.cognitive_state.goal_state),
                "task_state": dict(self.cognitive_state.task_state),
                "confidence_state": dict(self.cognitive_state.confidence_state),
                "environment_state": dict(self.cognitive_state.environment_state),
                "metadata": dict(self.cognitive_state.metadata),
            },
            "emotional_state": self.emotional_state.to_dict(),
            "retrieved_memories": [
                item.note.to_metadata() if hasattr(item, "note") and hasattr(item.note, "to_metadata") else str(item)
                for item in self.retrieved_memories
            ],
            # Safely handle possible None or objects without to_dict
            "active_plan": (self.active_plan.to_dict() if (self.active_plan is not None and hasattr(self.active_plan, "to_dict")) else self.active_plan),
            "reflection_signals": [
                signal.to_dict() if hasattr(signal, "to_dict") else str(signal)
                for signal in self.reflection_signals
            ],
            "confidence_score": self.confidence_score,
        }
