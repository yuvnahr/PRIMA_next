"""Shared typed outputs for the affect subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from affect.emotion_profile import EmotionProfile
from affect.pad_model import PADState

if TYPE_CHECKING:
    from state.emotional_state import EmotionalState


@dataclass(frozen=True, slots=True)
class ReflectionSignal:
    signal_type: str
    strength: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AffectUpdate:
    """Structured affect output consumed by other PRIMA subsystems."""

    profile: EmotionProfile
    pad_state: PADState
    emotional_state: EmotionalState
    retrieval_priors: dict[str, float]
    reflection_signals: tuple[ReflectionSignal, ...]
    memory_metadata: dict[str, Any]
    salience_score: float
    dissonance_score: float
    evolution: dict[str, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "pad_state": self.pad_state.to_dict(),
            "emotional_state": self.emotional_state.to_dict(),
            "retrieval_priors": dict(self.retrieval_priors),
            "reflection_signals": [
                {
                    "signal_type": signal.signal_type,
                    "strength": signal.strength,
                    "reason": signal.reason,
                    "metadata": dict(signal.metadata),
                }
                for signal in self.reflection_signals
            ],
            "memory_metadata": dict(self.memory_metadata),
            "salience_score": self.salience_score,
            "dissonance_score": self.dissonance_score,
            "evolution": dict(self.evolution),
            "timestamp": self.timestamp.isoformat(),
        }
