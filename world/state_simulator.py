"""Symbolic state simulator for plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from world.prediction_result import ActionPrediction, ConstraintPrediction, PredictionResult, StateTransition
from world.simulation_context import SimulationContext
from world.transition_predictor import TransitionPredictor
from world.world_state import WorldState, clamp01


@dataclass(slots=True)
class StateSimulator:
    """Predict plan consequences through deterministic symbolic rules."""

    transition_predictor: TransitionPredictor = field(default_factory=TransitionPredictor)

    def simulate(self, context: SimulationContext) -> PredictionResult:
        """Predict expected state, risk, failure probability, and constraints."""
        current_state = context.current_state
        action_predictions: list[ActionPrediction] = []
        transitions: list[StateTransition] = []
        constraint_predictions: list[ConstraintPrediction] = []
        warnings: list[str] = []

        actions = context.actions()
        if not actions:
            warnings.append("Plan has no actions; predicted state remains unchanged.")

        for action in actions:
            prediction, transition = self.transition_predictor.predict_action(
                action=action,
                current_state=current_state,
                constraints=context.constraints,
            )
            action_predictions.append(prediction)
            transitions.append(transition)
            constraint_predictions.extend(prediction.constraint_predictions)
            current_state = prediction.expected_state

        risk_score = self._aggregate_risk(action_predictions, constraint_predictions, context.plan_confidence())
        failure_probability = self._aggregate_failure(action_predictions, risk_score, context.plan_confidence())
        expected_state = current_state.merge_transition(
            {
                "confidence_state": {
                    "world_model_risk": risk_score,
                    "world_model_failure_probability": failure_probability,
                },
                "execution_state": {
                    "world_model_simulated": True,
                    "predicted_action_count": len(actions),
                },
                "metadata": {
                    "symbolic_world_model": True,
                    "deterministic": True,
                },
            }
        )

        warnings.extend(self._warnings(risk_score, failure_probability, constraint_predictions))
        return PredictionResult(
            initial_state=context.current_state,
            expected_state=expected_state,
            action_predictions=tuple(action_predictions),
            state_transitions=tuple(transitions),
            constraint_predictions=tuple(constraint_predictions),
            risk_score=risk_score,
            failure_probability=failure_probability,
            warnings=tuple(dict.fromkeys(warnings)),
            metadata={
                "plan_id": str(getattr(context.plan, "plan_id", "")),
                "symbolic": True,
                "deterministic": True,
                "ai_simulator": False,
            },
        )

    async def simulate_async(self, context: SimulationContext) -> PredictionResult:
        """Async-compatible wrapper around symbolic simulation."""
        return self.simulate(context)

    def simulate_from_inputs(
        self,
        current_state: WorldState | None = None,
        cognitive_state: Any | None = None,
        plan: Any | None = None,
        constraints: tuple[Any, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Build a simulation context and predict plan consequences."""
        context = SimulationContext.from_inputs(
            current_state=current_state,
            cognitive_state=cognitive_state,
            plan=plan,
            constraints=constraints,
            metadata=metadata,
        )
        return self.simulate(context)

    def _aggregate_risk(
        self,
        actions: list[ActionPrediction],
        constraints: list[ConstraintPrediction],
        plan_confidence: float,
    ) -> float:
        action_risk = fmean(action.risk_score for action in actions) if actions else 0.15
        constraint_risk = max((constraint.risk_contribution for constraint in constraints), default=0.0)
        confidence_gap = 1.0 - plan_confidence
        return round(clamp01(action_risk * 0.55 + constraint_risk * 0.25 + confidence_gap * 0.2), 6)

    def _aggregate_failure(
        self,
        actions: list[ActionPrediction],
        risk_score: float,
        plan_confidence: float,
    ) -> float:
        action_failure = fmean(action.failure_probability for action in actions) if actions else 0.2
        confidence_gap = 1.0 - plan_confidence
        return round(clamp01(action_failure * 0.55 + risk_score * 0.3 + confidence_gap * 0.15), 6)

    def _warnings(
        self,
        risk_score: float,
        failure_probability: float,
        constraints: list[ConstraintPrediction],
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        if risk_score >= 0.65:
            warnings.append("Predicted risk is high before execution.")
        if failure_probability >= 0.6:
            warnings.append("Predicted failure probability is elevated.")
        if any(constraint.violation_probability >= 0.55 for constraint in constraints):
            warnings.append("One or more constraints may be violated without mitigation.")
        return tuple(warnings)
