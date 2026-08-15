"""Typed post-execution validation signals and bounded correction advice."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from reasoning.models import AnswerResult
from reasoning.reflection_advisor import ReflectionAction, ReflectionAdvice, ReflectionEvent
from workflow.answer_generation import GenerationOutcome, GenerationResult
from workflow.execution_context import ExecutionContext
from workflow.workflow_state import WorkflowPhase


class OutputValidationSignal(str, Enum):
    """Observable post-execution failures eligible for correction."""

    OUTPUT_SCHEMA_FAILURE = "output_schema_failure"
    UNSUPPORTED_CLAIM = "unsupported_claim_evidence_mismatch"
    TOOL_FAILURE = "tool_failure"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    POLICY_VIOLATION = "policy_violation"


@dataclass(frozen=True, slots=True)
class OutputValidationResult:
    """Validation outcome and optional correction proposal."""

    valid: bool
    signals: tuple[OutputValidationSignal, ...] = ()
    advice: ReflectionAdvice | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the schema-versioned validation result."""
        return {
            "schema_version": "1.0",
            "valid": self.valid,
            "signals": [signal.value for signal in self.signals],
            "advice": self.advice.to_dict() if self.advice is not None else None,
        }


@dataclass(slots=True)
class OutputValidationController:
    """Validate generated answers and action results without executing corrections."""

    phase: WorkflowPhase = WorkflowPhase.OUTPUT_VALIDATION

    async def execute(self, context: ExecutionContext) -> OutputValidationResult:
        """Emit deterministic validation signals and one bounded proposal."""
        signals = self._signals(context)
        if not signals:
            return OutputValidationResult(valid=True)
        primary = signals[0]
        action = {
            OutputValidationSignal.OUTPUT_SCHEMA_FAILURE: ReflectionAction.REGENERATE,
            OutputValidationSignal.UNSUPPORTED_CLAIM: ReflectionAction.REVISE_QUERY,
            OutputValidationSignal.TOOL_FAILURE: ReflectionAction.RETRY_TRANSIENT_FAILURE,
            OutputValidationSignal.LOGICAL_INCONSISTENCY: ReflectionAction.REGENERATE,
            OutputValidationSignal.POLICY_VIOLATION: ReflectionAction.ABSTAIN,
        }[primary]
        event = {
            OutputValidationSignal.OUTPUT_SCHEMA_FAILURE: ReflectionEvent.OUTPUT_SCHEMA_FAILURE,
            OutputValidationSignal.UNSUPPORTED_CLAIM: ReflectionEvent.UNSUPPORTED_CLAIM,
            OutputValidationSignal.TOOL_FAILURE: ReflectionEvent.TOOL_FAILURE,
            OutputValidationSignal.LOGICAL_INCONSISTENCY: ReflectionEvent.LOGICAL_INCONSISTENCY,
            OutputValidationSignal.POLICY_VIOLATION: ReflectionEvent.POLICY_VIOLATION,
        }[primary]
        query = ""
        if action is ReflectionAction.REVISE_QUERY:
            query = f"{context.user_input.strip()} supporting evidence"
        return OutputValidationResult(
            valid=False,
            signals=signals,
            advice=ReflectionAdvice(
                trigger=True,
                action=action,
                suggested_query=query,
                correction_proposal=f"Correct post-execution signal: {primary.value}.",
                confidence=0.9,
                provenance={"component": "output_validator", "stage": "post_execution"},
                reason_code=primary.value,
                event=event,
            ),
        )

    def _signals(self, context: ExecutionContext) -> tuple[OutputValidationSignal, ...]:
        signals: list[OutputValidationSignal] = []
        generation = context.generation_result
        if isinstance(generation, GenerationResult):
            errors = " ".join(generation.errors).lower()
            if generation.outcome is GenerationOutcome.FAILED:
                if any(token in errors for token in ("json", "schema", "invalid", "empty", "evidence")):
                    signals.append(OutputValidationSignal.OUTPUT_SCHEMA_FAILURE)
                elif any(token in errors for token in ("inconsistent", "contradict")):
                    signals.append(OutputValidationSignal.LOGICAL_INCONSISTENCY)
                else:
                    signals.append(OutputValidationSignal.OUTPUT_SCHEMA_FAILURE)
            reasoning = context.reasoning_result
            factual = str(context.metadata.get("task_kind")) == "factual_qa"
            if (
                factual
                and generation.outcome is GenerationOutcome.ANSWERED
                and isinstance(reasoning, AnswerResult)
                and reasoning.evidence_references
                and not generation.selected_source_ids
            ):
                signals.append(OutputValidationSignal.UNSUPPORTED_CLAIM)

        action_payload = context.action_result if isinstance(context.action_result, dict) else {}
        serialized_action = str(action_payload).lower()
        status = str(action_payload.get("status", "")).lower()
        if status in {"failed", "partial"}:
            signals.append(OutputValidationSignal.TOOL_FAILURE)
        if status == "blocked" or "policy violation" in serialized_action:
            signals.append(OutputValidationSignal.POLICY_VIOLATION)
        return tuple(dict.fromkeys(signals))
