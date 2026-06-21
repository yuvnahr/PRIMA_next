"""Reflection result."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

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
    trigger_reasons: tuple[dict[str, object], ...] = field(default_factory=tuple)
    before_confidence: float = 0.0
    after_confidence: float = 0.0
    utility_score: float = 0.0

    @property
    def correction_applied(self) -> bool:
        """Return whether reflection produced meaningful confidence lift."""
        return self.should_reflect and self.utility_score >= 0.15

    def to_dict(self) -> dict[str, object]:
        """Serialize the result into JSON-friendly values."""
        return {
            "should_reflect": self.should_reflect,
            "trigger_score": self.trigger_score,
            "signals": [signal.to_dict() for signal in self.signals],
            "reflection_memory": self.reflection_memory.to_dict() if hasattr(self.reflection_memory, "to_dict") else None,
            "rules": [rule.to_dict() if hasattr(rule, "to_dict") else str(rule) for rule in self.rules],
            "confidence": asdict(self.confidence),
            "state_updates": dict(self.state_updates),
            "trigger_reasons": [dict(reason) for reason in self.trigger_reasons],
            "before_confidence": self.before_confidence,
            "after_confidence": self.after_confidence,
            "utility_score": self.utility_score,
            "correction_applied": self.correction_applied,
        }
