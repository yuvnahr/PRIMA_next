"""Simulation context for symbolic world-model prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from world.world_state import WorldState


@dataclass(frozen=True, slots=True)
class SimulationContext:
    """Inputs required for deterministic state simulation."""

    current_state: WorldState
    plan: Any
    constraints: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraints", tuple(self.constraints))

    @classmethod
    def from_inputs(
        cls,
        current_state: WorldState | None = None,
        cognitive_state: Any | None = None,
        plan: Any | None = None,
        constraints: tuple[Any, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> "SimulationContext":
        """Create context from either an explicit world state or cognitive state."""
        state = current_state or WorldState.from_cognitive_state(cognitive_state)
        plan_constraints = tuple(getattr(plan, "constraints", ()) or ())
        return cls(
            current_state=state,
            plan=plan,
            constraints=tuple(dict.fromkeys((*constraints, *plan_constraints))),
            metadata=metadata or {},
        )

    def actions(self) -> tuple[Any, ...]:
        """Return plan actions, accepting both typed plans and dict-shaped plans."""
        actions = getattr(self.plan, "actions", None)
        if actions is not None:
            return tuple(actions)
        if isinstance(self.plan, dict):
            return tuple(self.plan.get("actions", ()))
        return ()

    def plan_confidence(self) -> float:
        """Return the strongest confidence value exposed by the plan."""
        evaluation = getattr(self.plan, "evaluation", None)
        if evaluation is not None:
            try:
                return max(0.0, min(1.0, float(getattr(evaluation, "confidence", 0.0))))
            except (TypeError, ValueError):
                return 0.0
        if isinstance(self.plan, dict):
            evaluation_data = self.plan.get("evaluation") or {}
            try:
                return max(0.0, min(1.0, float(evaluation_data.get("confidence", 0.0))))
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the context into plain Python values."""
        return {
            "current_state": self.current_state.to_dict(),
            "plan_id": str(getattr(self.plan, "plan_id", "")),
            "action_count": len(self.actions()),
            "constraint_count": len(self.constraints),
            "metadata": dict(self.metadata),
        }
