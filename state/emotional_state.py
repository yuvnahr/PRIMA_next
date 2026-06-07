"""Persistent emotional state model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from affect.pad_model import PADState


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class EmotionalState:
    """Persistent emotional state consumed by PRIMA cognitive state."""

    current_pad: PADState = field(default_factory=PADState)
    baseline_pad: PADState = field(default_factory=PADState)
    dominant_emotion: str = "neutral"
    emotional_momentum: dict[str, float] = field(default_factory=dict)
    emotional_stability: float = 1.0
    emotional_volatility: float = 0.0
    confidence: float = 0.0
    last_update_time: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_pad": self.current_pad.to_dict(),
            "baseline_pad": self.baseline_pad.to_dict(),
            "dominant_emotion": self.dominant_emotion,
            "emotional_momentum": dict(self.emotional_momentum),
            "emotional_stability": self.emotional_stability,
            "emotional_volatility": self.emotional_volatility,
            "confidence": self.confidence,
            "last_update_time": self.last_update_time.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmotionalState":
        last_update = data.get("last_update_time")
        if isinstance(last_update, str):
            parsed_update = datetime.fromisoformat(last_update)
        elif isinstance(last_update, datetime):
            parsed_update = last_update
        else:
            parsed_update = utc_now()

        return cls(
            current_pad=PADState.from_dict(data.get("current_pad", {})),
            baseline_pad=PADState.from_dict(data.get("baseline_pad", {})),
            dominant_emotion=str(data.get("dominant_emotion", "neutral")),
            emotional_momentum=dict(data.get("emotional_momentum", {})),
            emotional_stability=float(data.get("emotional_stability", 1.0)),
            emotional_volatility=float(data.get("emotional_volatility", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            last_update_time=parsed_update,
        )
