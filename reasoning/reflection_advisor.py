"""Typed boundary for bounded reflection advice."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class ReflectionEvent(str, Enum):
    """Pre- and post-execution conditions that may request correction."""

    LOW_CONFIDENCE = "low_confidence"
    NO_NEW_EVIDENCE = "no_new_evidence"
    DUPLICATE_QUERY = "duplicate_query"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    RETRIEVAL_FAILURE = "retrieval_failure"
    PLAN_CONSTRAINT_VIOLATION = "plan_constraint_violation"
    OUTPUT_SCHEMA_FAILURE = "output_schema_failure"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    TOOL_FAILURE = "tool_failure"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    POLICY_VIOLATION = "policy_violation"


class ReflectionAction(str, Enum):
    """Bounded correction actions accepted by the workflow."""

    REVISE_QUERY = "revise_query"
    BROADEN_QUERY = "broaden_query"
    PIVOT_ENTITY = "pivot_entity"
    REPLAN = "replan"
    RETRY_TRANSIENT_FAILURE = "retry_transient_failure"
    REGENERATE = "regenerate"
    ASK_USER = "ask_user"
    ABSTAIN = "abstain"
    NO_ACTION = "no_action"

    CONTINUE_WITH_REVISED_QUERY = "revise_query"
    RESOLVE_CONTRADICTION = "replan"
    ASK_CLARIFICATION = "ask_user"


@dataclass(frozen=True, slots=True)
class ReflectionAdvice:
    """Typed correction proposal that cannot control orchestration directly."""

    trigger: bool
    action: ReflectionAction = ReflectionAction.NO_ACTION
    suggested_query: str = ""
    correction_proposal: str = ""
    confidence: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)
    reason_code: str = ""
    event: ReflectionEvent | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", ReflectionAction(self.action))
        object.__setattr__(self, "event", ReflectionEvent(self.event) if self.event is not None else None)
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))

    @property
    def should_intervene(self) -> bool:
        """Compatibility alias for the typed trigger flag."""
        return self.trigger

    def contains_evaluation_leakage(self) -> bool:
        """Detect gold/evaluation information at the correction boundary."""
        payload = " ".join(
            (
                self.suggested_query,
                self.correction_proposal,
                self.reason_code,
                str(self.provenance),
                str(self.metadata),
            )
        ).lower()
        return any(
            token in payload
            for token in (
                "gold",
                "ground_truth",
                "ground truth",
                "expected_answer",
                "expected answer",
                "correct_answer",
                "supporting_fact",
                "evaluation result",
                "retrieval scoring",
                "rerank",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize advice without exposing hidden reasoning."""
        return {
            "trigger": self.trigger,
            "action": self.action.value,
            "suggested_query": self.suggested_query,
            "correction_proposal": self.correction_proposal,
            "confidence": self.confidence,
            "provenance": dict(self.provenance),
            "reason_code": self.reason_code,
            "event": self.event.value if self.event is not None else None,
            "metadata": dict(self.metadata),
        }


class ReflectionAdvisor(Protocol):
    """Advice-only boundary; callers retain control of all execution."""

    def advise(self, event: ReflectionEvent, state: Any, *, query: str, stop_reason: str) -> ReflectionAdvice:
        """Return bounded advice for one observed reflection event."""
