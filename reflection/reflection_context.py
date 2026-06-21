"""Primary input context for adaptive reflection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory.retrieval.retrieval_confidence import RetrievalConfidence
from reflection.reflection_signal import ReflectionSignal
from reflection.reflection_types import FailureType
from state.cognitive_state import CognitiveState
from state.emotional_state import EmotionalState


@dataclass(frozen=True, slots=True)
class ReflectionContext:
    query: str
    retrieved_memories: tuple[Any, ...] = ()
    retrieval_confidence: RetrievalConfidence | None = None
    affect_confidence: float | None = None
    cognitive_state: CognitiveState | None = None
    emotional_state: EmotionalState | None = None
    reflection_history: tuple[str, ...] = ()
    failure_type: FailureType | None = None
    failure_metadata: dict[str, Any] = field(default_factory=dict)
    affect_signals: tuple[Any, ...] = ()

    def state_snapshot(self) -> dict[str, Any]:
        if self.cognitive_state is None:
            return {}
        return {
            "goal_state": dict(self.cognitive_state.goal_state),
            "task_state": dict(self.cognitive_state.task_state),
            "confidence_state": dict(self.cognitive_state.confidence_state),
            "emotional_state": self.cognitive_state.emotional_state.to_dict(),
            "environment_state": dict(self.cognitive_state.environment_state),
        }

    def signal_inputs(self) -> tuple[ReflectionSignal, ...]:
        return tuple(signal for signal in self.affect_signals if isinstance(signal, ReflectionSignal))
