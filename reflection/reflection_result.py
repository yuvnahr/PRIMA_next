"""Reflection result."""

from __future__ import annotations

from dataclasses import dataclass

from reflection.reflection_confidence import ReflectionConfidence
from reflection.reflection_memory import ReflectionMemory, Rule
from reflection.reflection_signal import ReflectionSignal


@dataclass(frozen=True, slots=True)
class ReflectionResult:
    should_reflect: bool
    trigger_score: float
    signals: tuple[ReflectionSignal, ...]
    reflection_memory: ReflectionMemory | None
    rules: tuple[Rule, ...]
    confidence: ReflectionConfidence
    state_updates: dict[str, object]
