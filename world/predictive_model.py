"""Deterministic symbolic prediction rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from world.prediction_result import ConstraintPrediction
from world.world_state import WorldState, clamp01


def _value(value: Any) -> str:
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _action_metadata(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        return dict(action.get("metadata", {}))
    metadata = getattr(action, "metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


@dataclass(frozen=True, slots=True)
class SymbolicRule:
    """One deterministic state-transition rule."""

    action_type: str
    transition: dict[str, Any]
    risk_delta: float
    failure_delta: float
    rationale: str


@dataclass(slots=True)
class SymbolicPredictiveModel:
    """Rule-based predictor for plan-action consequences."""

    rules: dict[str, SymbolicRule] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rules:
            self.rules = self.default_rules()

    @staticmethod
    def default_rules() -> dict[str, SymbolicRule]:
        """Return deterministic rules keyed by planning action type."""
        rules = (
            SymbolicRule(
                "review_state",
                {"execution_state": {"state_reviewed": True, "last_symbolic_step": "review_state"}},
                0.03,
                0.02,
                "Reviewing state lowers unknowns without external effects.",
            ),
            SymbolicRule(
                "integrate_memory",
                {
                    "execution_state": {"memory_integrated": True, "last_symbolic_step": "integrate_memory"},
                    "confidence_state": {"memory_grounding_predicted": True},
                },
                0.12,
                0.08,
                "Memory integration improves grounding but can inherit retrieval ambiguity.",
            ),
            SymbolicRule(
                "apply_affective_priors",
                {"execution_state": {"affect_priors_applied": True, "last_symbolic_step": "apply_affective_priors"}},
                0.08,
                0.05,
                "Affective priors modify planning without controlling downstream subsystems.",
            ),
            SymbolicRule(
                "resolve_uncertainty",
                {"execution_state": {"uncertainty_addressed": True, "last_symbolic_step": "resolve_uncertainty"}},
                0.16,
                0.12,
                "Uncertainty handling reduces unsupported assumptions but signals fragile context.",
            ),
            SymbolicRule(
                "request_reflection",
                {
                    "execution_state": {"reflection_checkpoint_predicted": True, "last_symbolic_step": "request_reflection"},
                    "confidence_state": {"reflection_needed_predicted": True},
                },
                0.1,
                0.08,
                "Reflection checkpoint mitigates risk before execution.",
            ),
            SymbolicRule(
                "plan_state_transition",
                {"execution_state": {"state_transition_planned": True, "last_symbolic_step": "plan_state_transition"}},
                0.14,
                0.1,
                "State transition planning estimates downstream changes symbolically.",
            ),
            SymbolicRule(
                "prepare_response",
                {"execution_state": {"response_prepared": True, "last_symbolic_step": "prepare_response"}},
                0.18,
                0.14,
                "Preparing a response is closest to execution and carries residual grounding risk.",
            ),
        )
        return {rule.action_type: rule for rule in rules}

    def transition_for(self, action: Any, state: WorldState) -> tuple[dict[str, Any], str]:
        """Return the symbolic transition and rationale for an action."""
        action_type = self.action_type(action)
        rule = self.rules.get(action_type)
        if rule is None:
            return (
                {"execution_state": {"unknown_action_predicted": action_type}},
                "Unknown action type receives a conservative symbolic transition.",
            )

        transition = _deep_copy_transition(rule.transition)
        metadata = _action_metadata(action)
        if metadata:
            transition.setdefault("metadata", {})[f"action_{self.action_id(action)}"] = metadata
        confidence = state.confidence()
        if confidence:
            transition.setdefault("confidence_state", {})["pre_execution_confidence"] = confidence
        return transition, rule.rationale

    def risk_for(self, action: Any, state: WorldState, constraints: tuple[Any, ...]) -> float:
        """Return deterministic action risk."""
        action_type = self.action_type(action)
        rule = self.rules.get(action_type)
        base = rule.risk_delta if rule else 0.3
        confidence_gap = 1.0 - state.confidence()
        constraint_pressure = min(0.25, len(constraints) * 0.025)
        external_tool_pressure = 0.25 if self.requires_external_tool(action) else 0.0
        return round(clamp01(base + confidence_gap * 0.2 + constraint_pressure + external_tool_pressure), 6)

    def failure_probability_for(self, action: Any, state: WorldState, constraints: tuple[Any, ...]) -> float:
        """Return deterministic action failure probability."""
        action_type = self.action_type(action)
        rule = self.rules.get(action_type)
        base = rule.failure_delta if rule else 0.28
        confidence_gap = 1.0 - state.confidence()
        hard_constraint_pressure = min(0.3, self.required_constraint_count(constraints) * 0.04)
        return round(clamp01(base + confidence_gap * 0.25 + hard_constraint_pressure), 6)

    def constraint_predictions(self, action: Any, constraints: tuple[Any, ...], state: WorldState) -> tuple[ConstraintPrediction, ...]:
        """Predict constraint violation pressure for an action."""
        predictions: list[ConstraintPrediction] = []
        action_type = self.action_type(action)
        for constraint in constraints:
            description = str(getattr(constraint, "description", constraint))
            constraint_id = str(getattr(constraint, "constraint_id", description[:48]))
            required = bool(getattr(constraint, "required", True))
            severity = 0.8 if required else 0.35
            violation_probability = self._constraint_violation_probability(description, action_type, state)
            predictions.append(
                ConstraintPrediction(
                    constraint_id=constraint_id,
                    description=description,
                    violation_probability=violation_probability,
                    severity=severity,
                    mitigation=self._mitigation(description, action_type),
                    metadata={"action_type": action_type, "required": required},
                )
            )
        return tuple(predictions)

    def action_type(self, action: Any) -> str:
        """Return a normalized action type."""
        if isinstance(action, dict):
            return _value(action.get("action_type", "unknown"))
        return _value(getattr(action, "action_type", "unknown"))

    def action_id(self, action: Any) -> str:
        """Return a stable action identifier when available."""
        if isinstance(action, dict):
            return str(action.get("action_id", self.action_type(action)))
        return str(getattr(action, "action_id", self.action_type(action)))

    def requires_external_tool(self, action: Any) -> bool:
        """Return whether an action declares external tool requirements."""
        if isinstance(action, dict):
            metadata = dict(action.get("metadata", {}))
            return bool(action.get("requires_external_tool", metadata.get("requires_external_tool", False)))
        metadata = _action_metadata(action)
        return bool(getattr(action, "requires_external_tool", metadata.get("requires_external_tool", False)))

    def required_constraint_count(self, constraints: tuple[Any, ...]) -> int:
        """Count required constraints."""
        return sum(1 for constraint in constraints if bool(getattr(constraint, "required", True)))

    def _constraint_violation_probability(self, description: str, action_type: str, state: WorldState) -> float:
        lowered = description.lower()
        probability = 0.08
        if "must not execute external tools" in lowered:
            probability += 0.4 if action_type not in {"review_state", "request_reflection"} else 0.05
        if "must not invoke language models" in lowered:
            probability += 0.05
        if "uncertainty" in lowered or "unsupported" in lowered:
            probability += (1.0 - state.confidence()) * 0.3
        if "reflection" in lowered and action_type != "request_reflection":
            probability += 0.15
        if "workflow" in lowered:
            probability += 0.04
        return round(clamp01(probability), 6)

    def _mitigation(self, description: str, action_type: str) -> str:
        lowered = description.lower()
        if "external tools" in lowered:
            return "Keep this as an intent and let workflow-owned action dispatch decide."
        if "language models" in lowered:
            return "Use symbolic state and plan metadata only."
        if "reflection" in lowered and action_type != "request_reflection":
            return "Insert or preserve a reflection checkpoint before execution."
        if "uncertainty" in lowered or "unsupported" in lowered:
            return "Carry uncertainty into confidence state and avoid unsupported claims."
        return "Preserve the constraint during workflow-owned execution."


def _deep_copy_transition(transition: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in transition.items():
        copied[key] = dict(value) if isinstance(value, dict) else value
    return copied
