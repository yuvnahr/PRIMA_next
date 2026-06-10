"""Goal selection heuristics for pure planning."""

from __future__ import annotations

from typing import Any

from planning.plan import PlanGoal, clamp01
from planning.planning_context import PlanningContext
from planning.planning_types import GoalPriority


class GoalSelector:
    """Select a goal from persistent state and the current objective."""

    def select_goal(self, context: PlanningContext) -> PlanGoal:
        """Return the primary goal for the plan."""
        description, source = self._goal_description(context)
        priority = self._priority(context)
        confidence = self._confidence(context)
        return PlanGoal.create(
            description=description,
            priority=priority,
            confidence=confidence,
            source=source,
            metadata={
                "retrieved_memory_count": len(context.retrieved_memories),
                "max_reflection_severity": context.max_reflection_severity,
                "dominant_emotion": context.affective_state.get("dominant_emotion", "neutral"),
            },
        )

    def _goal_description(self, context: PlanningContext) -> tuple[str, str]:
        for key in ("active_goal", "current_goal", "objective", "goal"):
            value = context.goal_state.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), f"cognitive_state.{key}"
        return context.objective.strip() or "maintain cognitive continuity", "user_input"

    def _priority(self, context: PlanningContext) -> GoalPriority:
        explicit = context.goal_state.get("priority")
        if isinstance(explicit, str):
            normalized = explicit.strip().lower()
            if normalized in {priority.value for priority in GoalPriority}:
                return GoalPriority(normalized)
        if isinstance(explicit, int | float):
            if explicit >= 0.9:
                return GoalPriority.CRITICAL
            if explicit >= 0.65:
                return GoalPriority.HIGH
            if explicit <= 0.25:
                return GoalPriority.LOW

        urgency = self._urgency_score(context)
        if urgency >= 0.85:
            return GoalPriority.CRITICAL
        if urgency >= 0.55:
            return GoalPriority.HIGH
        if urgency <= 0.2:
            return GoalPriority.LOW
        return GoalPriority.MEDIUM

    def _confidence(self, context: PlanningContext) -> float:
        explicit = _float_or_none(context.confidence_state.get("confidence"))
        if explicit is not None:
            return clamp01(explicit)

        profile_confidence = _float_or_none(context.affective_state.get("profile_confidence")) or 0.0
        memory_support = context.retrieval_confidence if context.retrieved_memories else 0.25
        signal_penalty = context.max_reflection_severity * 0.25
        confidence = memory_support * 0.45 + profile_confidence * 0.25 + 0.35 - signal_penalty
        return round(clamp01(confidence), 6)

    def _urgency_score(self, context: PlanningContext) -> float:
        objective = context.objective.lower()
        urgent_terms = ("urgent", "immediately", "critical", "failure", "blocked", "emergency")
        text_urgency = 0.35 if any(term in objective for term in urgent_terms) else 0.0
        low_confidence_pressure = max(0.0, 0.45 - context.retrieval_confidence)
        affect_pressure = float(context.affective_state.get("dissonance_score", 0.0)) * 0.4
        signal_pressure = context.max_reflection_severity * 0.5
        return clamp01(text_urgency + low_confidence_pressure + affect_pressure + signal_pressure)


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
