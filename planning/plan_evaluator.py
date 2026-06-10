"""Plan simulation and evaluation."""

from __future__ import annotations

from statistics import fmean

from planning.plan import Plan, PlanEvaluation, PlanSimulation, PlanStepSimulation, clamp01
from planning.planning_context import PlanningContext
from planning.planning_types import ActionStatus, ActionType


class PlanEvaluator:
    """Evaluate plans through deterministic local simulation."""

    def simulate(self, plan: Plan, context: PlanningContext) -> PlanSimulation:
        """Simulate plan execution without performing side effects."""
        steps: list[PlanStepSimulation] = []
        requires_reflection = False

        for action in plan.actions:
            risk = self._action_risk(action.action_type, context)
            state_changes = {
                "planned_action": action.action_type.value,
                "expected_output": action.expected_output,
            }
            notes: tuple[str, ...] = ()
            if action.action_type == ActionType.REQUEST_REFLECTION:
                requires_reflection = True
                state_changes["reflection_checkpoint"] = "requested"
                notes = ("workflow_reflection_phase_should_evaluate_before_action",)
            elif action.action_type == ActionType.RESOLVE_UNCERTAINTY:
                state_changes["uncertainty_policy"] = "surface_or_reduce_uncertainty"
                notes = ("no_unsupported_memory_assumptions",)

            steps.append(
                PlanStepSimulation(
                    action_id=action.action_id,
                    predicted_status=ActionStatus.SIMULATED,
                    predicted_state_changes=state_changes,
                    risk_score=risk,
                    notes=notes,
                )
            )

        predicted_risk = fmean(step.risk_score for step in steps) if steps else 0.0
        memory_factor = context.retrieval_confidence if context.retrieved_memories else 0.2
        signal_penalty = context.max_reflection_severity * 0.2
        predicted_confidence = clamp01(plan.goal.confidence * 0.45 + memory_factor * 0.35 + (1.0 - predicted_risk) * 0.2)
        predicted_confidence = round(clamp01(predicted_confidence - signal_penalty), 6)

        return PlanSimulation(
            steps=tuple(steps),
            predicted_confidence=predicted_confidence,
            predicted_risk=round(clamp01(predicted_risk), 6),
            requires_reflection=requires_reflection,
            metadata={
                "simulated_action_count": len(steps),
                "pure_reasoning": True,
            },
        )

    def evaluate(
        self,
        plan: Plan,
        context: PlanningContext,
        simulation: PlanSimulation | None = None,
    ) -> PlanEvaluation:
        """Evaluate a plan and determine whether replanning is needed."""
        simulation = simulation or self.simulate(plan, context)
        reasons: list[str] = []

        if context.retrieval_confidence < 0.4:
            reasons.append("low_retrieval_confidence")
        if not context.retrieved_memories:
            reasons.append("memory_context_absent")
        if context.max_reflection_severity >= 0.45:
            reasons.append("reflection_signal_pressure")
        if float(context.affective_state.get("dissonance_score", 0.0)) >= 0.5:
            reasons.append("affective_dissonance")

        uncertainty = self._uncertainty(context)
        should_replan = bool(simulation.blocked_constraints) or (
            uncertainty >= 0.85 and not simulation.requires_reflection
        )
        confidence = clamp01(simulation.predicted_confidence * 0.7 + (1.0 - uncertainty) * 0.3)
        return PlanEvaluation(
            confidence=round(confidence, 6),
            uncertainty=round(uncertainty, 6),
            risk_score=simulation.predicted_risk,
            should_replan=should_replan,
            reasons=tuple(reasons),
            metadata={
                "requires_reflection": simulation.requires_reflection,
                "retrieved_memory_count": len(context.retrieved_memories),
            },
        )

    def _action_risk(self, action_type: ActionType, context: PlanningContext) -> float:
        base = {
            ActionType.REVIEW_STATE: 0.05,
            ActionType.INTEGRATE_MEMORY: 0.15,
            ActionType.APPLY_AFFECTIVE_PRIORS: 0.12,
            ActionType.RESOLVE_UNCERTAINTY: 0.18,
            ActionType.REQUEST_REFLECTION: 0.08,
            ActionType.PLAN_STATE_TRANSITION: 0.16,
            ActionType.PREPARE_RESPONSE: 0.2,
        }[action_type]
        low_confidence_penalty = max(0.0, 0.4 - context.retrieval_confidence) * 0.35
        signal_penalty = context.max_reflection_severity * 0.25
        return round(clamp01(base + low_confidence_penalty + signal_penalty), 6)

    def _uncertainty(self, context: PlanningContext) -> float:
        memory_gap = 0.25 if not context.retrieved_memories else 0.0
        confidence_gap = 1.0 - context.retrieval_confidence
        dissonance = float(context.affective_state.get("dissonance_score", 0.0))
        uncertainty = confidence_gap * 0.35 + context.retrieval_ambiguity * 0.2
        uncertainty += context.max_reflection_severity * 0.25 + dissonance * 0.15 + memory_gap
        return clamp01(uncertainty)
