"""Action-level transition prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from world.prediction_result import ActionPrediction, StateTransition
from world.predictive_model import SymbolicPredictiveModel
from world.world_state import WorldState


@dataclass(slots=True)
class TransitionPredictor:
    """Predict symbolic state transitions for plan actions."""

    model: SymbolicPredictiveModel = field(default_factory=SymbolicPredictiveModel)

    def predict_action(
        self,
        action: Any,
        current_state: WorldState,
        constraints: tuple[Any, ...],
    ) -> tuple[ActionPrediction, StateTransition]:
        """Predict the expected state after one symbolic action."""
        delta, rationale = self.model.transition_for(action, current_state)
        expected_state = current_state.merge_transition(delta)
        action_id = self.model.action_id(action)
        action_type = self.model.action_type(action)
        constraint_predictions = self.model.constraint_predictions(action, constraints, current_state)
        constraint_risk = max((item.risk_contribution for item in constraint_predictions), default=0.0)
        risk = max(self.model.risk_for(action, current_state, constraints), constraint_risk)
        failure = max(self.model.failure_probability_for(action, current_state, constraints), constraint_risk * 0.85)
        prediction = ActionPrediction(
            action_id=action_id,
            action_type=action_type,
            expected_state=expected_state,
            risk_score=risk,
            failure_probability=failure,
            constraint_predictions=constraint_predictions,
            rationale=rationale,
            metadata={"symbolic": True, "deterministic": True},
        )
        transition = StateTransition(
            action_id=action_id,
            before_state=current_state,
            after_state=expected_state,
            delta=delta,
            rationale=rationale,
        )
        return prediction, transition
