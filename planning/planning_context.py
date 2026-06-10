"""Planning context adapters for workflow-routed subsystem outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from planning.plan import clamp01


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        return dict(data) if isinstance(data, Mapping) else {}
    return {}


def _enum_value(value: Any) -> str:
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


@dataclass(frozen=True, slots=True)
class PlanningMemory:
    """Memory summary consumed by the planner."""

    memory_id: str
    content: str
    score: float
    memory_type: str = "unknown"
    salience_score: float = 0.0
    keywords: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", clamp01(self.score))
        object.__setattr__(self, "salience_score", clamp01(self.salience_score))
        object.__setattr__(self, "keywords", tuple(str(keyword) for keyword in self.keywords))

    @classmethod
    def from_retrieval_result(cls, result: Any) -> "PlanningMemory":
        """Create a planning memory from a retrieval result-like object."""
        note = getattr(result, "note", None)
        note_id = str(getattr(note, "id", "memory_unknown"))
        keywords = getattr(note, "keywords", ())
        metadata = {
            "strategy_scores": _as_dict(getattr(result, "strategy_scores", {})),
            "explanation": _as_dict(getattr(result, "explanation", {})),
            "context": _as_dict(getattr(note, "context", {})),
            "affective_state": _as_dict(getattr(note, "affective_state", {})),
        }
        return cls(
            memory_id=note_id,
            content=str(getattr(note, "content", "")),
            score=clamp01(float(getattr(result, "score", 0.0))),
            memory_type=_enum_value(getattr(note, "memory_type", "unknown")),
            salience_score=clamp01(float(getattr(note, "salience_score", 0.0))),
            keywords=tuple(str(keyword) for keyword in keywords),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the memory summary into plain Python values."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "score": self.score,
            "memory_type": self.memory_type,
            "salience_score": self.salience_score,
            "keywords": list(self.keywords),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PlanningReflectionSignal:
    """Reflection signal normalized for planning."""

    signal_type: str
    severity: float
    confidence: float
    source: str
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", clamp01(self.severity))
        object.__setattr__(self, "confidence", clamp01(self.confidence))

    @classmethod
    def from_signal(cls, signal: Any) -> "PlanningReflectionSignal":
        """Normalize an affect or reflection signal-like object."""
        severity = getattr(signal, "severity", getattr(signal, "strength", 0.0))
        confidence = getattr(signal, "confidence", severity)
        return cls(
            signal_type=_enum_value(getattr(signal, "signal_type", "unknown")),
            severity=clamp01(float(severity)),
            confidence=clamp01(float(confidence)),
            source=str(getattr(signal, "source", "unknown")),
            reason=str(getattr(signal, "reason", "")),
            metadata=_as_dict(getattr(signal, "metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the signal into plain Python values."""
        return {
            "signal_type": self.signal_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "source": self.source,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PlanningContext:
    """Planner input assembled by workflow from subsystem outputs."""

    objective: str
    goal_state: dict[str, Any] = field(default_factory=dict)
    task_state: dict[str, Any] = field(default_factory=dict)
    confidence_state: dict[str, Any] = field(default_factory=dict)
    environment_state: dict[str, Any] = field(default_factory=dict)
    emotional_state: dict[str, Any] = field(default_factory=dict)
    retrieved_memories: tuple[PlanningMemory, ...] = ()
    retrieval_confidence: float = 0.0
    retrieval_ambiguity: float = 0.0
    affective_priors: dict[str, float] = field(default_factory=dict)
    reflection_signals: tuple[PlanningReflectionSignal, ...] = ()
    affective_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "retrieved_memories", tuple(self.retrieved_memories))
        object.__setattr__(self, "retrieval_confidence", clamp01(self.retrieval_confidence))
        object.__setattr__(self, "retrieval_ambiguity", clamp01(self.retrieval_ambiguity))
        object.__setattr__(
            self,
            "affective_priors",
            {str(key): clamp01(value) for key, value in self.affective_priors.items()},
        )
        object.__setattr__(self, "reflection_signals", tuple(self.reflection_signals))

    @classmethod
    def from_subsystem_outputs(
        cls,
        objective: str,
        cognitive_state: Any | None = None,
        retrieval_response: Any | None = None,
        affect_update: Any | None = None,
        reflection_result: Any | None = None,
        reflection_signals: Sequence[Any] = (),
        metadata: dict[str, Any] | None = None,
    ) -> "PlanningContext":
        """Build a planning context from workflow-owned subsystem outputs."""
        confidence = getattr(retrieval_response, "confidence", None)
        retrieval_results = tuple(getattr(retrieval_response, "results", ()) or ())
        normalized_signals = [
            PlanningReflectionSignal.from_signal(signal)
            for signal in tuple(getattr(affect_update, "reflection_signals", ()) or ())
        ]
        normalized_signals.extend(PlanningReflectionSignal.from_signal(signal) for signal in reflection_signals)
        normalized_signals.extend(
            PlanningReflectionSignal.from_signal(signal)
            for signal in tuple(getattr(reflection_result, "signals", ()) or ())
        )

        profile = getattr(affect_update, "profile", None)
        affective_state = {
            "dominant_emotion": getattr(profile, "dominant_emotion", "neutral"),
            "profile_confidence": clamp01(float(getattr(profile, "confidence", 0.0))),
            "salience_score": clamp01(float(getattr(affect_update, "salience_score", 0.0))),
            "dissonance_score": clamp01(float(getattr(affect_update, "dissonance_score", 0.0))),
            "evolution": _as_dict(getattr(affect_update, "evolution", {})),
        }

        emotional_state = _as_dict(getattr(cognitive_state, "emotional_state", None))
        if not emotional_state:
            emotional_state = _as_dict(getattr(affect_update, "emotional_state", None))

        return cls(
            objective=objective,
            goal_state=_as_dict(getattr(cognitive_state, "goal_state", {})),
            task_state=_as_dict(getattr(cognitive_state, "task_state", {})),
            confidence_state=_as_dict(getattr(cognitive_state, "confidence_state", {})),
            environment_state=_as_dict(getattr(cognitive_state, "environment_state", {})),
            emotional_state=emotional_state,
            retrieved_memories=tuple(PlanningMemory.from_retrieval_result(result) for result in retrieval_results),
            retrieval_confidence=clamp01(float(getattr(confidence, "confidence", 0.0))),
            retrieval_ambiguity=clamp01(float(getattr(confidence, "ambiguity_score", 0.0))),
            affective_priors=dict(getattr(affect_update, "retrieval_priors", {}) or {}),
            reflection_signals=tuple(normalized_signals),
            affective_state=affective_state,
            metadata=metadata or {},
        )

    @property
    def max_reflection_severity(self) -> float:
        """Return the highest normalized reflection signal severity."""
        if not self.reflection_signals:
            return 0.0
        return max(signal.severity for signal in self.reflection_signals)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the context into plain Python values."""
        return {
            "objective": self.objective,
            "goal_state": dict(self.goal_state),
            "task_state": dict(self.task_state),
            "confidence_state": dict(self.confidence_state),
            "environment_state": dict(self.environment_state),
            "emotional_state": dict(self.emotional_state),
            "retrieved_memories": [memory.to_dict() for memory in self.retrieved_memories],
            "retrieval_confidence": self.retrieval_confidence,
            "retrieval_ambiguity": self.retrieval_ambiguity,
            "affective_priors": dict(self.affective_priors),
            "reflection_signals": [signal.to_dict() for signal in self.reflection_signals],
            "affective_state": dict(self.affective_state),
            "metadata": dict(self.metadata),
        }
