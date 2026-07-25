"""Small typed boundary for optional reflection advice."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

class ReflectionEvent(str, Enum):
    NO_NEW_EVIDENCE = "no_new_evidence"
    DUPLICATE_QUERY = "duplicate_query"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    RETRIEVAL_FAILURE = "retrieval_failure"

class ReflectionAction(str, Enum):
    CONTINUE_WITH_REVISED_QUERY = "continue_with_revised_query"
    BROADEN_QUERY = "broaden_query"
    PIVOT_ENTITY = "pivot_entity"
    RESOLVE_CONTRADICTION = "resolve_contradiction"
    RETRY_TRANSIENT_FAILURE = "retry_transient_failure"
    ASK_CLARIFICATION = "ask_clarification"
    ABSTAIN = "abstain"
    NO_ACTION = "no_action"

@dataclass(frozen=True, slots=True)
class ReflectionAdvice:
    should_intervene: bool
    event: ReflectionEvent
    action: ReflectionAction = ReflectionAction.NO_ACTION
    confidence: float = 0.0
    suggested_query: str = ""
    reason_code: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

class ReflectionAdvisor(Protocol):
    def advise(self, event: ReflectionEvent, state: Any, *, query: str, stop_reason: str) -> ReflectionAdvice:
        """Return advice only; the controller retains all control flow."""
