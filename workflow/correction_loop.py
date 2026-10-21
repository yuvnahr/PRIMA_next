"""Bounded acceptance and audit records for workflow corrections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm.request_budget import RequestBudget
from reasoning.information_need_generator import InformationNeedGenerator
from reasoning.models import AnswerResult, EvidenceState, ReasoningRequest, SufficiencyDecision, SufficiencyStatus
from reasoning.reflection_advisor import ReflectionAction, ReflectionAdvice


@dataclass(frozen=True, slots=True)
class CorrectionBudget:
    """Hard limits for one workflow execution's correction loop."""

    max_retrieval_retries: int = 1
    max_replans: int = 1
    max_reflection_interventions: int = 1
    max_generations: int = 2
    advice_confidence_threshold: float = 0.6

    def __post_init__(self) -> None:
        for name in (
            "max_retrieval_retries",
            "max_replans",
            "max_reflection_interventions",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.max_generations < 1:
            raise ValueError("max_generations must be at least one.")
        if not 0.0 <= self.advice_confidence_threshold <= 1.0:
            raise ValueError("advice_confidence_threshold must be between 0 and 1.")

    def to_dict(self) -> dict[str, float | int]:
        """Serialize configured loop limits for diagnostics."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(slots=True)
class CorrectionAttempt:
    """Actual before/after snapshots for one accepted or rejected proposal."""

    attempt_index: int
    stage: str
    advice: ReflectionAdvice
    accepted: bool
    decision_reason: str
    before_query: str
    after_query: str
    before_plan: dict[str, Any] | None
    after_plan: dict[str, Any] | None
    before_answer: str
    after_answer: str
    before_confidence: float
    after_confidence: float
    confidence_delta: float = 0.0
    utility: float = 0.0
    proposal_accepted: bool = False
    executed: bool = False
    validated: bool | None = None
    rejected_output: dict[str, Any] | None = None
    before_generation: Any = field(default=None, repr=False)
    before_generation_count: int = 0
    before_retrieval_count: int = 0
    before_prompt_evidence: list[dict[str, Any]] = field(default_factory=list, repr=False)
    before_reasoning: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize one schema-versioned correction record."""
        return {
            "schema_version": "2.0",
            "attempt_index": self.attempt_index,
            "stage": self.stage,
            "advice": self.advice.to_dict(),
            "accepted": self.accepted,
            "proposal_accepted": self.proposal_accepted, "executed": self.executed, "validated": self.validated,
            "rejected_output": self.rejected_output, "utility_interpretation": "heuristic_change_score",
            "decision_reason": self.decision_reason,
            "before_query": self.before_query,
            "after_query": self.after_query,
            "before_plan": self.before_plan,
            "after_plan": self.after_plan,
            "before_answer": self.before_answer,
            "after_answer": self.after_answer,
            "before_confidence": self.before_confidence,
            "after_confidence": self.after_confidence,
            "confidence_delta": self.confidence_delta,
            "utility": self.utility,
        }


@dataclass(slots=True)
class CorrectionLoop:
    """Validate advice, enforce budgets, and retain attempt snapshots."""

    budget: CorrectionBudget = field(default_factory=CorrectionBudget)

    def review(self, context: Any, advice: ReflectionAdvice, stage: str) -> CorrectionAttempt:
        """Accept or reject advice without executing its requested action."""
        query = str(context.metadata.get("retrieval_query_override", context.user_input))
        candidate = " ".join(advice.suggested_query.split())
        action = advice.action
        if action in _QUERY_ACTIONS and isinstance(context.reasoning_result, AnswerResult):
            state = EvidenceState(ReasoningRequest(context.user_input), evidence_items=list(context.reasoning_result.evidence_references))
            decision = SufficiencyDecision(context.reasoning_result.status, context.reasoning_result.confidence, False)
            candidate = InformationNeedGenerator().supported_query(state, decision, candidate) or ""
        reason = self._rejection_reason(context, advice, query, candidate)
        accepted = not reason
        if accepted:
            reason = "accepted"
            self._consume_budget(context, action, candidate)
        attempt = CorrectionAttempt(
            attempt_index=len(context.correction_attempts) + 1,
            stage=stage,
            advice=advice,
            accepted=False,
            proposal_accepted=accepted,
            before_generation=context.generation_result,
            before_generation_count=int(context.metadata.get("generation_count", 0)),
            before_retrieval_count=int(context.metadata.get("evidence_acquisition_attempts", 0)),
            before_prompt_evidence=list(context.prompt_evidence), before_reasoning=context.reasoning_result,
            decision_reason=reason,
            before_query=query,
            after_query=candidate if accepted and candidate else query,
            before_plan=_plan_snapshot(context.plan),
            after_plan=_plan_snapshot(context.plan),
            before_answer=_answer_snapshot(context.generation_result),
            after_answer=_answer_snapshot(context.generation_result),
            before_confidence=_confidence_snapshot(context),
            after_confidence=_confidence_snapshot(context),
        )
        context.correction_attempts.append(attempt)
        return attempt

    def complete(self, context: Any) -> None:
        """Populate after-snapshots from the latest real workflow state."""
        query = str(context.metadata.get("retrieval_query_override", context.user_input))
        plan = _plan_snapshot(context.plan)
        answer = _answer_snapshot(context.generation_result)
        confidence = _confidence_snapshot(context)
        for attempt in context.correction_attempts:
            if not attempt.proposal_accepted or attempt.validated is not None:
                continue
            attempt.executed = (int(context.metadata.get("generation_count", 0)) > attempt.before_generation_count
                or int(context.metadata.get("evidence_acquisition_attempts", 0)) > attempt.before_retrieval_count
                or _plan_snapshot(context.plan) != attempt.before_plan)
            if not attempt.executed and getattr(getattr(context.workflow_state, "current_phase", None), "value", "") != "output":
                continue
            valid = bool(getattr(context.output_validation, "valid", False))
            candidate_generation = context.generation_result
            valid = valid and not (getattr(candidate_generation, "errors", ())
                                  or getattr(candidate_generation, "validation_errors", ())
                                  or getattr(candidate_generation, "failure_kind", None))
            if str(context.metadata.get("task_kind")) == "factual_qa":
                generation = context.generation_result
                selected = set(getattr(generation, "selected_source_ids", ()))
                supplied = {row["memory_id"] for row in context.prompt_evidence}
                outcome = getattr(getattr(generation, "outcome", None), "value", "")
                valid = valid and (outcome == "abstained" or (
                    outcome == "answered" and bool(selected) and selected <= supplied
                    and bool(getattr(context.reasoning_result, "evidence_references", ()))))
            attempt.validated = valid
            attempt.accepted = bool(attempt.executed and valid)
            if not attempt.accepted:
                candidate = context.generation_result
                attempt.rejected_output = candidate.to_dict() if hasattr(candidate, "to_dict") else {"text": answer}
                attempt.rejected_output["prompt_evidence"] = list(context.prompt_evidence)
                attempt.decision_reason = "invalid_corrected_output" if attempt.executed else "correction_not_executed"
                if attempt.before_generation is not None:
                    context.generation_result = attempt.before_generation
                    context.prompt_evidence = list(attempt.before_prompt_evidence)
                    context.reasoning_result = attempt.before_reasoning
                    answer = _answer_snapshot(context.generation_result)
            else:
                attempt.decision_reason = "validated_evidence_and_output"
            attempt.after_query = query
            attempt.after_plan = plan
            attempt.after_answer = answer
            attempt.after_confidence = confidence
            attempt.confidence_delta = round(confidence - attempt.before_confidence, 6)
            changed = (
                attempt.before_query != attempt.after_query
                or attempt.before_plan != attempt.after_plan
                or attempt.before_answer != attempt.after_answer
            )
            attempt.utility = round(max(0.0, attempt.confidence_delta, 0.25 if changed else 0.0), 6)

    def _rejection_reason(
        self,
        context: Any,
        advice: ReflectionAdvice,
        query: str,
        candidate: str,
    ) -> str:
        budget: RequestBudget | None = getattr(context, "request_budget", None)
        if not context.metadata.get("reflection_enabled", True):
            return "reflection_off"
        gate_decision = getattr(getattr(context, "execution_decision", None), "decision", None)
        if str(gate_decision) in {"abstain", "ask_for_clarification"}:
            return "gate_terminal_decision"
        if budget is not None:
            if budget.remaining_seconds <= 0:
                return "time_budget"
            if advice.action is ReflectionAction.REGENERATE:
                if budget.denial_reason():
                    return budget.denial_reason()
                reasoning = getattr(context, "reasoning_result", None)
                if (isinstance(reasoning, AnswerResult) and reasoning.status is not SufficiencyStatus.SUFFICIENT
                        and budget.bounded_answer_attempts):
                    return "answer_attempt_exhausted"
            if advice.action in _QUERY_ACTIONS and (budget.retrieval_time_exhausted
                    or budget.retrieval_calls >= int(context.metadata.get("max_retrieval_calls", 3))):
                return "retrieval_budget_exhausted"
        if not advice.trigger or advice.action is ReflectionAction.NO_ACTION:
            return "no_intervention"
        if advice.contains_evaluation_leakage():
            return "evaluation_leakage"
        if advice.confidence < self.budget.advice_confidence_threshold:
            return "low_advice_confidence"
        signature = (advice.action.value, candidate.lower())
        if signature in context.metadata.setdefault("correction_signatures", set()):
            return "duplicate_advice"
        if int(context.metadata.get("reflection_intervention_count", 0)) >= self.budget.max_reflection_interventions:
            return "reflection_budget_exhausted"
        if advice.action in _QUERY_ACTIONS:
            if not candidate:
                return "missing_query"
            attempted = {
                query.lower(),
                *(str(item).lower() for item in context.metadata.get("correction_queries", ())),
            }
            if candidate.lower() in attempted:
                return "duplicate_advice"
            retries = int(context.metadata.get("correction_retrieval_retries", 0)) + int(
                context.metadata.get("retrieval_retry_count", 0)
            )
            if retries >= self.budget.max_retrieval_retries:
                return "retrieval_budget_exhausted"
        if advice.action is ReflectionAction.RETRY_TRANSIENT_FAILURE:
            retries = int(context.metadata.get("correction_retrieval_retries", 0)) + int(
                context.metadata.get("retrieval_retry_count", 0)
            )
            if retries >= self.budget.max_retrieval_retries:
                return "retrieval_budget_exhausted"
        if (
            advice.action is ReflectionAction.REPLAN
            and int(context.metadata.get("correction_replans", 0)) >= self.budget.max_replans
        ):
            return "replan_budget_exhausted"
        if (
            advice.action is ReflectionAction.REGENERATE
            and int(context.metadata.get("generation_count", 0)) >= self.budget.max_generations
        ):
            return "generation_budget_exhausted"
        return ""

    def _consume_budget(self, context: Any, action: ReflectionAction, candidate: str) -> None:
        context.metadata["reflection_intervention_count"] = int(
            context.metadata.get("reflection_intervention_count", 0)
        ) + 1
        signature = (action.value, candidate.lower())
        signatures = context.metadata.setdefault("correction_signatures", set())
        signatures.add(signature)
        if action in _QUERY_ACTIONS or action is ReflectionAction.RETRY_TRANSIENT_FAILURE:
            context.metadata["correction_retrieval_retries"] = int(
                context.metadata.get("correction_retrieval_retries", 0)
            ) + 1
            context.metadata.setdefault("correction_queries", []).append(candidate)
        elif action is ReflectionAction.REPLAN:
            context.metadata["correction_replans"] = int(context.metadata.get("correction_replans", 0)) + 1


_QUERY_ACTIONS = {
    ReflectionAction.REVISE_QUERY,
    ReflectionAction.BROADEN_QUERY,
    ReflectionAction.PIVOT_ENTITY,
}


def _plan_snapshot(plan: Any) -> dict[str, Any] | None:
    if plan is None:
        return None
    value = plan.to_dict() if hasattr(plan, "to_dict") else plan
    return dict(value) if isinstance(value, dict) else {"value": str(value)}


def _answer_snapshot(generation: Any) -> str:
    return str(getattr(generation, "text", "") or "")


def _confidence_snapshot(context: Any) -> float:
    uncertainty = getattr(context.uncertainty, "confidence", None)
    if uncertainty is not None:
        return round(float(uncertainty), 6)
    reasoning = getattr(context.reasoning_result, "confidence", None)
    return round(float(reasoning or 0.0), 6)
