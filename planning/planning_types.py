"""Typed constants for PRIMA-NEXT planning."""

from __future__ import annotations

from enum import Enum


class GoalPriority(str, Enum):
    """Planner priority assigned to the selected goal."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    """Pure reasoning action categories.

    These are intended execution intents only. They never execute tools or call
    model providers directly.
    """

    REVIEW_STATE = "review_state"
    INTEGRATE_MEMORY = "integrate_memory"
    APPLY_AFFECTIVE_PRIORS = "apply_affective_priors"
    RESOLVE_UNCERTAINTY = "resolve_uncertainty"
    REQUEST_REFLECTION = "request_reflection"
    PLAN_STATE_TRANSITION = "plan_state_transition"
    PREPARE_RESPONSE = "prepare_response"


class ActionStatus(str, Enum):
    """Planning-time action status."""

    PENDING = "pending"
    SIMULATED = "simulated"
    BLOCKED = "blocked"


class ConstraintType(str, Enum):
    """Constraint domains attached to plans and actions."""

    EXECUTION = "execution"
    MEMORY = "memory"
    STATE = "state"
    AFFECT = "affect"
    REFLECTION = "reflection"
    POLICY = "policy"
    SAFETY = "safety"


class ExecutionIntentType(str, Enum):
    """High-level intent produced for the workflow action phase."""

    RESPOND = "respond"
    RESPOND_AFTER_REFLECTION = "respond_after_reflection"
    REQUEST_CLARIFICATION = "request_clarification"
    NO_OP = "no_op"


class PlanStatus(str, Enum):
    """Lifecycle status for immutable planning outputs."""

    DRAFT = "draft"
    EVALUATED = "evaluated"
    NEEDS_REPLAN = "needs_replan"


class ReplanReason(str, Enum):
    """Canonical reasons for replanning."""

    LOW_CONFIDENCE = "low_confidence"
    CONSTRAINT_VIOLATION = "constraint_violation"
    REFLECTION_SIGNAL = "reflection_signal"
    GOAL_CHANGE = "goal_change"
    EXECUTION_FEEDBACK = "execution_feedback"
