"""Action and execution-intent selection for planning."""

from __future__ import annotations

from planning.plan import ExecutionIntent, PlanAction, PlanConstraint, PlanGoal, clamp01
from planning.planning_context import PlanningContext
from planning.planning_types import ActionType, ConstraintType, ExecutionIntentType, GoalPriority


class ActionSelector:
    """Select pure planning actions from context and constraints."""

    def select_actions(
        self,
        context: PlanningContext,
        goal: PlanGoal,
        constraints: tuple[PlanConstraint, ...],
    ) -> tuple[PlanAction, ...]:
        """Return ordered action proposals for a plan."""
        constraint_ids = tuple(constraint.constraint_id for constraint in constraints)
        actions = [
            PlanAction.create(
                ActionType.REVIEW_STATE,
                "Review persistent goal, task, confidence, emotional, and environment state.",
                ("goal_state", "task_state", "confidence_state", "emotional_state", "environment_state"),
                "state_assessment",
                constraint_ids,
            )
        ]

        if context.retrieved_memories:
            actions.append(
                PlanAction.create(
                    ActionType.INTEGRATE_MEMORY,
                    "Integrate retrieved memories as grounding context for the selected goal.",
                    tuple(memory.memory_id for memory in context.retrieved_memories),
                    "memory_grounding",
                    _constraint_ids_by_type(constraints, ConstraintType.MEMORY),
                    metadata={"memory_count": len(context.retrieved_memories)},
                )
            )
        else:
            actions.append(
                PlanAction.create(
                    ActionType.RESOLVE_UNCERTAINTY,
                    "Record memory absence and avoid unsupported assumptions.",
                    ("retrieval_confidence",),
                    "memory_gap_assessment",
                    _constraint_ids_by_type(constraints, ConstraintType.MEMORY),
                )
            )

        if context.affective_priors:
            actions.append(
                PlanAction.create(
                    ActionType.APPLY_AFFECTIVE_PRIORS,
                    "Apply affective priors as planning modifiers without controlling retrieval directly.",
                    tuple(sorted(context.affective_priors)),
                    "affect_adjusted_plan",
                    _constraint_ids_by_type(constraints, ConstraintType.AFFECT),
                    metadata={"priors": dict(context.affective_priors)},
                )
            )

        if self._needs_reflection_checkpoint(context):
            actions.append(
                PlanAction.create(
                    ActionType.REQUEST_REFLECTION,
                    "Route uncertainty or reflection signals to the workflow reflection checkpoint.",
                    ("reflection_signals", "retrieval_confidence"),
                    "reflection_checkpoint_intent",
                    _constraint_ids_by_type(constraints, ConstraintType.REFLECTION),
                    metadata={
                        "max_reflection_severity": context.max_reflection_severity,
                        "retrieval_confidence": context.retrieval_confidence,
                    },
                )
            )

        actions.append(
            PlanAction.create(
                ActionType.PLAN_STATE_TRANSITION,
                "Predict state transition requirements for the action phase.",
                ("state_assessment", "memory_grounding", "affect_adjusted_plan"),
                "state_transition_prediction",
                _constraint_ids_by_type(constraints, ConstraintType.STATE),
            )
        )
        actions.append(
            PlanAction.create(
                ActionType.PREPARE_RESPONSE,
                "Prepare execution intent for workflow-owned action dispatch.",
                (goal.goal_id,),
                "execution_intent",
                constraint_ids,
            )
        )
        return tuple(actions)

    def select_execution_intent(
        self,
        context: PlanningContext,
        goal: PlanGoal,
        actions: tuple[PlanAction, ...],
    ) -> ExecutionIntent:
        """Return the high-level execution intent for the plan."""
        has_reflection_checkpoint = any(action.action_type == ActionType.REQUEST_REFLECTION for action in actions)
        if has_reflection_checkpoint:
            intent_type = ExecutionIntentType.RESPOND_AFTER_REFLECTION
            rationale = "Reflection checkpoint requested before final action."
        elif not context.retrieved_memories and context.retrieval_confidence <= 0.15:
            intent_type = ExecutionIntentType.REQUEST_CLARIFICATION
            rationale = "Memory context is absent and retrieval confidence is too low for grounded response."
        else:
            intent_type = ExecutionIntentType.RESPOND
            rationale = "State, memory, and affect inputs are sufficient for a workflow-owned response."

        priority_bonus = 0.1 if goal.priority in {GoalPriority.HIGH, GoalPriority.CRITICAL} else 0.0
        confidence = clamp01(goal.confidence + priority_bonus - context.max_reflection_severity * 0.15)
        return ExecutionIntent(
            intent_type=intent_type,
            objective=goal.description,
            rationale=rationale,
            confidence=round(confidence, 6),
            requires_external_tool=False,
            metadata={"action_count": len(actions)},
        )

    def _needs_reflection_checkpoint(self, context: PlanningContext) -> bool:
        return (
            context.max_reflection_severity >= 0.45
            or context.retrieval_confidence < 0.4
            or float(context.affective_state.get("dissonance_score", 0.0)) >= 0.5
        )


def _constraint_ids_by_type(
    constraints: tuple[PlanConstraint, ...],
    constraint_type: ConstraintType,
) -> tuple[str, ...]:
    return tuple(
        constraint.constraint_id
        for constraint in constraints
        if constraint.constraint_type in {constraint_type, ConstraintType.EXECUTION, ConstraintType.POLICY}
    )
