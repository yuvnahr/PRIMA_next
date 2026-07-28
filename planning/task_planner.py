"""Pure task planner for PRIMA-NEXT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from planning.action_selector import ActionSelector
from planning.goal_selector import GoalSelector
from planning.plan import (
    Plan,
    PlanAction,
    PlanConstraint,
    PlanEvaluation,
    PlanSimulation,
    stable_id,
)
from planning.plan_evaluator import PlanEvaluator
from planning.planning_context import PlanningContext
from planning.planning_types import ActionType, ConstraintType


@dataclass(slots=True)
class TaskPlanner:
    """Build, simulate, evaluate, and replan structured pure-reasoning plans."""

    goal_selector: GoalSelector = field(default_factory=GoalSelector)
    action_selector: ActionSelector = field(default_factory=ActionSelector)
    evaluator: PlanEvaluator = field(default_factory=PlanEvaluator)

    def create_plan(self, context: PlanningContext) -> Plan:
        """Create a structured plan from memory, state, affect, and signals."""
        return self._build_plan(context=context, revision=0, previous_plan_id=None, replan_reason=None)

    async def create_plan_async(self, context: PlanningContext) -> Plan:
        """Async-compatible wrapper around create_plan."""
        return self.create_plan(context)

    def replan(self, context: PlanningContext, prior_plan: Plan, reason: str) -> Plan:
        """Create a revised plan linked to a previous plan."""
        return self._build_plan(
            context=context,
            revision=prior_plan.revision + 1,
            previous_plan_id=prior_plan.plan_id,
            replan_reason=reason,
        )

    async def replan_async(self, context: PlanningContext, prior_plan: Plan, reason: str) -> Plan:
        """Async-compatible wrapper around replan."""
        return self.replan(context, prior_plan, reason)

    def simulate_execution(self, plan: Plan, context: PlanningContext) -> PlanSimulation:
        """Simulate a plan without executing tools or external effects."""
        return self.evaluator.simulate(plan, context)

    def evaluate_plan(
        self,
        plan: Plan,
        context: PlanningContext,
        simulation: PlanSimulation | None = None,
    ) -> PlanEvaluation:
        """Evaluate a plan's confidence, uncertainty, and replanning need."""
        return self.evaluator.evaluate(plan, context, simulation)

    def _build_plan(
        self,
        context: PlanningContext,
        revision: int,
        previous_plan_id: str | None,
        replan_reason: str | None,
    ) -> Plan:
        goal = self.goal_selector.select_goal(context)
        constraints = self._constraints(context, replan_reason)
        actions = self.action_selector.select_actions(context, goal, constraints)
        if replan_reason:
            actions = (self._correction_action(replan_reason, constraints), *actions)
        execution_intent = self.action_selector.select_execution_intent(context, goal, actions)
        plan = Plan(
            plan_id=stable_id(
                "plan",
                context.objective,
                goal.goal_id,
                tuple(action.action_id for action in actions),
                revision,
                previous_plan_id or "root",
            ),
            objective=context.objective,
            goal=goal,
            actions=actions,
            execution_intent=execution_intent,
            constraints=constraints,
            revision=revision,
            previous_plan_id=previous_plan_id,
            metadata={
                "replan_reason": replan_reason,
                "pure_reasoning": True,
                "direct_tool_execution": False,
                "direct_llm_invocation": False,
            },
        )
        simulation = self.evaluator.simulate(plan, context)
        evaluation = self.evaluator.evaluate(plan, context, simulation)
        return plan.with_simulation_and_evaluation(simulation, evaluation)

    def _constraints(self, context: PlanningContext, replan_reason: str | None) -> tuple[PlanConstraint, ...]:
        constraints = [
            PlanConstraint.create(
                ConstraintType.EXECUTION,
                "Planner must not execute external tools.",
                metadata={"hard": True},
            ),
            PlanConstraint.create(
                ConstraintType.EXECUTION,
                "Planner must not invoke language models.",
                metadata={"hard": True},
            ),
            PlanConstraint.create(
                ConstraintType.POLICY,
                "Workflow layer owns orchestration and execution dispatch.",
                metadata={"hard": True},
            ),
        ]

        if context.retrieval_confidence < 0.4 or not context.retrieved_memories:
            constraints.append(
                PlanConstraint.create(
                    ConstraintType.MEMORY,
                    "Surface retrieval uncertainty and avoid unsupported memory claims.",
                )
            )
        if context.affective_priors:
            constraints.append(
                PlanConstraint.create(
                    ConstraintType.AFFECT,
                    "Use affective priors only as planning modifiers.",
                )
            )
        if context.max_reflection_severity >= 0.45:
            constraints.append(
                PlanConstraint.create(
                    ConstraintType.REFLECTION,
                    "Route severe reflection signals to workflow reflection before action.",
                )
            )
        if replan_reason:
            constraints.append(
                PlanConstraint.create(
                    ConstraintType.POLICY,
                    f"Replan because {replan_reason}; preserve previous plan lineage.",
                )
            )

        constraints.extend(self._state_constraints(context))
        return _dedupe_constraints(tuple(constraints))

    def _state_constraints(self, context: PlanningContext) -> tuple[PlanConstraint, ...]:
        state_constraints: list[PlanConstraint] = []
        for section_name, section in (
            ("goal_state", context.goal_state),
            ("task_state", context.task_state),
            ("environment_state", context.environment_state),
        ):
            for key in ("constraints", "planning_constraints"):
                for item in _constraint_items(section.get(key)):
                    state_constraints.append(
                        PlanConstraint.create(
                            ConstraintType.STATE,
                            str(item),
                            metadata={"source": f"{section_name}.{key}"},
                        )
                    )
        return tuple(state_constraints)

    def _correction_action(self, reason: str, constraints: tuple[PlanConstraint, ...]) -> PlanAction:
        policy_constraints = tuple(
            constraint.constraint_id
            for constraint in constraints
            if constraint.constraint_type in {ConstraintType.POLICY, ConstraintType.REFLECTION, ConstraintType.MEMORY}
        )
        return PlanAction.create(
            ActionType.RESOLVE_UNCERTAINTY,
            f"Address replanning reason before continuing: {reason}.",
            ("prior_plan", "reflection_signals", "state_constraints"),
            "corrected_plan_assumptions",
            policy_constraints,
            metadata={"replan_reason": reason},
        )


def _constraint_items(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple | set):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


def _dedupe_constraints(constraints: tuple[PlanConstraint, ...]) -> tuple[PlanConstraint, ...]:
    seen: set[str] = set()
    deduped: list[PlanConstraint] = []
    for constraint in constraints:
        if constraint.constraint_id in seen:
            continue
        seen.add(constraint.constraint_id)
        deduped.append(constraint)
    return tuple(deduped)
