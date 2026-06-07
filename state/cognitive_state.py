"""Minimal cognitive state placeholder for subsystem integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from state.emotional_state import EmotionalState


@dataclass(slots=True)
class CognitiveState:
    """Container consumed by affect, memory, and reflection without ownership."""

    emotional_state: EmotionalState = field(default_factory=EmotionalState)
    goal_state: dict[str, Any] = field(default_factory=dict)
    task_state: dict[str, Any] = field(default_factory=dict)
    confidence_state: dict[str, Any] = field(default_factory=dict)
    environment_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
