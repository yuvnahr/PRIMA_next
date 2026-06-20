"""Structured plan domain models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any

from planning.planning_types import (
    ActionStatus,
    ActionType,
    ConstraintType,
    ExecutionIntentType,
    GoalPriority,
    PlanStatus,
)


def clamp01(value: float) -> float:
    """Clamp a score into the normalized [0, 1] interval."""
    return max(0.0, min(1.0, float(value)))


def stable_id(prefix: str, *parts: object) -> str:
    """Create a deterministic identifier from stable plan fields."""
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


@dataclass(frozen=True, slots=True)
class PlanGoal:
    """Planner-selected goal derived from state and user objective."""

    goal_id: str
    description: str
    priority: GoalPriority
    confidence: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "priority", GoalPriority(self.priority))
        object.__setattr__(self, "confidence", clamp01(self.confidence))

    @classmethod
    def create(
        cls,
        description: str,
        priority: GoalPriority,
        confidence: float,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> PlanGoal:
        """Create a deterministic goal."""
        return cls(
            goal_id=stable_id("goal", description, priority.value, source),
            description=description,
            priority=priority,
            confidence=confidence,
            source=source,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the goal into plain Python values."""
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "priority": self.priority.value,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PlanConstraint:
    """A planning constraint produced from policy, state, memory, or affect."""

    constraint_id: str
    constraint_type: ConstraintType
    description: str
    required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraint_type", ConstraintType(self.constraint_type))

    @classmethod
    def create(
        cls,
        constraint_type: ConstraintType,
        description: str,
        required: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> PlanConstraint:
        """Create a deterministic constraint."""
        return cls(
            constraint_id=stable_id("constraint", constraint_type.value, description),
            constraint_type=constraint_type,
            description=description,
            required=required,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the constraint into plain Python values."""
        return {
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type.value,
            "description": self.description,
            "required": self.required,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PlanAction:
    """A pure planning action proposal.

    The action records what the workflow may later dispatch. It does not perform
    external effects by itself.
    """

    action_id: str
    action_type: ActionType
    description: str
    inputs: tuple[str, ...]
    expected_output: str
    constraints: tuple[str, ...] = ()
    status: ActionStatus = ActionStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_type", ActionType(self.action_type))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "status", ActionStatus(self.status))

    @classmethod
    def create(
        cls,
        action_type: ActionType,
        description: str,
        inputs: tuple[str, ...],
        expected_output: str,
        constraints: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> PlanAction:
        """Create a deterministic action proposal."""
        return cls(
            action_id=stable_id("action", action_type.value, description, inputs, expected_output),
            action_type=action_type,
            description=description,
            inputs=inputs,
            expected_output=expected_output,
            constraints=constraints,
            metadata=metadata or {},
        )

    def with_status(self, status: ActionStatus) -> PlanAction:
        """Return this action with an updated planning status."""
        return replace(self, status=status)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the action into plain Python values."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "description": self.description,
            "inputs": list(self.inputs),
            "expected_output": self.expected_output,
            "constraints": list(self.constraints),
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ExecutionIntent:
    """Planner intent consumed by the workflow action phase."""

    intent_type: ExecutionIntentType
    objective: str
    rationale: str
    confidence: float
    requires_external_tool: bool = False
    tool_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_type", ExecutionIntentType(self.intent_type))
        object.__setattr__(self, "confidence", clamp01(self.confidence))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the intent into plain Python values."""
        return {
            "intent_type": self.intent_type.value,
            "objective": self.objective,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "requires_external_tool": self.requires_external_tool,
            "tool_name": self.tool_name,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PlanStepSimulation:
    """Predicted result for one planned action."""

    action_id: str
    predicted_status: ActionStatus
    predicted_state_changes: dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicted_status", ActionStatus(self.predicted_status))
        object.__setattr__(self, "risk_score", clamp01(self.risk_score))
        object.__setattr__(self, "notes", tuple(self.notes))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the step simulation into plain Python values."""
        return {
            "action_id": self.action_id,
            "predicted_status": self.predicted_status.value,
            "predicted_state_changes": dict(self.predicted_state_changes),
            "risk_score": self.risk_score,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class PlanSimulation:
    """Pure simulation trace for a plan."""

    steps: tuple[PlanStepSimulation, ...]
    predicted_confidence: float
    predicted_risk: float
    requires_reflection: bool
    blocked_constraints: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "predicted_confidence", clamp01(self.predicted_confidence))
        object.__setattr__(self, "predicted_risk", clamp01(self.predicted_risk))
        object.__setattr__(self, "blocked_constraints", tuple(self.blocked_constraints))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the simulation into plain Python values."""
        return {
            "steps": [step.to_dict() for step in self.steps],
            "predicted_confidence": self.predicted_confidence,
            "predicted_risk": self.predicted_risk,
            "requires_reflection": self.requires_reflection,
            "blocked_constraints": list(self.blocked_constraints),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PlanEvaluation:
    """Evaluation summary for deciding whether a plan is usable."""

    confidence: float
    uncertainty: float
    risk_score: float
    should_replan: bool
    reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", clamp01(self.confidence))
        object.__setattr__(self, "uncertainty", clamp01(self.uncertainty))
        object.__setattr__(self, "risk_score", clamp01(self.risk_score))
        object.__setattr__(self, "reasons", tuple(self.reasons))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the evaluation into plain Python values."""
        return {
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "risk_score": self.risk_score,
            "should_replan": self.should_replan,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Plan:
    """Immutable structured plan produced by the pure planning layer."""

    plan_id: str
    objective: str
    goal: PlanGoal
    actions: tuple[PlanAction, ...]
    execution_intent: ExecutionIntent
    constraints: tuple[PlanConstraint, ...]
    status: PlanStatus = PlanStatus.DRAFT
    simulation: PlanSimulation | None = None
    evaluation: PlanEvaluation | None = None
    revision: int = 0
    previous_plan_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "status", PlanStatus(self.status))

    def with_simulation_and_evaluation(
        self,
        simulation: PlanSimulation,
        evaluation: PlanEvaluation,
    ) -> Plan:
        """Return this plan with simulation and evaluation attached."""
        status = PlanStatus.NEEDS_REPLAN if evaluation.should_replan else PlanStatus.EVALUATED
        return replace(self, simulation=simulation, evaluation=evaluation, status=status)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan into plain Python values."""
        return {
            "plan_id": self.plan_id,
            "objective": self.objective,
            "goal": self.goal.to_dict(),
            "actions": [action.to_dict() for action in self.actions],
            "execution_intent": self.execution_intent.to_dict(),
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "status": self.status.value,
            "simulation": self.simulation.to_dict() if self.simulation else None,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "revision": self.revision,
            "previous_plan_id": self.previous_plan_id,
            "metadata": dict(self.metadata),
        }
