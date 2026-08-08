"""One-way adapter from bounded failures to usable reflection advice."""

from __future__ import annotations

from dataclasses import dataclass

from reasoning.reflection_advisor import ReflectionAction, ReflectionAdvice, ReflectionEvent
from reflection.reflection_context import ReflectionContext
from reflection.reflection_engine import ReflectionEngine
from reflection.reflection_types import FailureType


@dataclass(slots=True)
class ReasoningReflectionAdapter:
    """Translate reflection-engine signals into workflow-safe corrections."""

    engine: ReflectionEngine

    def advise(self, event: ReflectionEvent, state: object, *, query: str, stop_reason: str) -> ReflectionAdvice:
        """Return deterministic advice that reasoning and workflow can apply."""
        failure_type = {
            ReflectionEvent.CONTRADICTORY_EVIDENCE: FailureType.MEMORY_FAILURE,
            ReflectionEvent.RETRIEVAL_FAILURE: FailureType.RETRIEVAL_FAILURE,
            ReflectionEvent.PLAN_CONSTRAINT_VIOLATION: FailureType.PLANNING_FAILURE,
            ReflectionEvent.TOOL_FAILURE: FailureType.TOOL_FAILURE,
            ReflectionEvent.POLICY_VIOLATION: FailureType.GOAL_CONFLICT,
            ReflectionEvent.UNSUPPORTED_CLAIM: FailureType.HALLUCINATION_RISK,
        }.get(event, FailureType.REASONING_FAILURE)
        result = self.engine.evaluate(
            ReflectionContext(
                query=query,
                retrieved_memories=tuple(getattr(state, "evidence_items", ())),
                failure_type=failure_type,
                failure_metadata={"reason": stop_reason, "severity": 1.0, "reasoning_event": True},
            )
        )
        return self.advice_for_result(event, result, query=query, stop_reason=stop_reason)

    def advice_for_result(
        self,
        event: ReflectionEvent,
        result: object,
        *,
        query: str,
        stop_reason: str,
    ) -> ReflectionAdvice:
        """Translate an already-computed reflection result without reevaluation."""
        action = {
            ReflectionEvent.CONTRADICTORY_EVIDENCE: ReflectionAction.REVISE_QUERY,
            ReflectionEvent.DUPLICATE_QUERY: ReflectionAction.REVISE_QUERY,
            ReflectionEvent.NO_NEW_EVIDENCE: ReflectionAction.BROADEN_QUERY,
            ReflectionEvent.RETRIEVAL_FAILURE: ReflectionAction.RETRY_TRANSIENT_FAILURE,
            ReflectionEvent.PLAN_CONSTRAINT_VIOLATION: ReflectionAction.REPLAN,
            ReflectionEvent.OUTPUT_SCHEMA_FAILURE: ReflectionAction.REGENERATE,
            ReflectionEvent.UNSUPPORTED_CLAIM: ReflectionAction.REVISE_QUERY,
            ReflectionEvent.TOOL_FAILURE: ReflectionAction.RETRY_TRANSIENT_FAILURE,
            ReflectionEvent.LOGICAL_INCONSISTENCY: ReflectionAction.REGENERATE,
            ReflectionEvent.POLICY_VIOLATION: ReflectionAction.ABSTAIN,
            ReflectionEvent.LOW_CONFIDENCE: ReflectionAction.BROADEN_QUERY,
        }[event]
        suggested_query = ""
        if action in {
            ReflectionAction.REVISE_QUERY,
            ReflectionAction.BROADEN_QUERY,
            ReflectionAction.PIVOT_ENTITY,
        }:
            suggested_query = f"{query.strip()} additional independent context"
        elif action is ReflectionAction.RETRY_TRANSIENT_FAILURE:
            suggested_query = query.strip()
        return ReflectionAdvice(
            trigger=bool(getattr(result, "should_reflect", False)),
            action=action,
            suggested_query=suggested_query,
            correction_proposal=f"Apply {action.value} for {stop_reason}.",
            confidence=float(getattr(getattr(result, "confidence", None), "overall_confidence", 0.0)),
            provenance={"component": "reflection_engine", "trigger_score": getattr(result, "trigger_score", 0.0)},
            reason_code=(
                str(getattr(getattr(result, "reflection_memory", None), "source_failure", ""))
                or event.value
            ),
            event=event,
        )
