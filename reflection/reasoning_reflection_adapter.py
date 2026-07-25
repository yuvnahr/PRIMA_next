"""One-way adapter from bounded reasoning failures to reflection advice."""

from __future__ import annotations

from dataclasses import dataclass
from reflection.reflection_context import ReflectionContext
from reflection.reflection_engine import ReflectionEngine
from reflection.reflection_types import FailureType
from reasoning.reflection_advisor import ReflectionAction, ReflectionAdvice, ReflectionEvent

@dataclass(slots=True)
class ReasoningReflectionAdapter:
    engine: ReflectionEngine

    def advise(self, event: ReflectionEvent, state: object, *, query: str, stop_reason: str) -> ReflectionAdvice:
        failure_type = {ReflectionEvent.CONTRADICTORY_EVIDENCE: FailureType.MEMORY_FAILURE, ReflectionEvent.RETRIEVAL_FAILURE: FailureType.RETRIEVAL_FAILURE}.get(event, FailureType.REASONING_FAILURE)
        result = self.engine.evaluate(ReflectionContext(query=query, retrieved_memories=tuple(getattr(state, "evidence_items", ()),), failure_type=failure_type, failure_metadata={"reason": stop_reason, "severity": 1.0, "reasoning_event": True}))
        action = ReflectionAction.ABSTAIN if event is ReflectionEvent.CONTRADICTORY_EVIDENCE else ReflectionAction.NO_ACTION
        return ReflectionAdvice(should_intervene=result.should_reflect, event=event, action=action, confidence=result.confidence.overall_confidence, reason_code=failure_type.value, metadata={"trigger_score": result.trigger_score})
