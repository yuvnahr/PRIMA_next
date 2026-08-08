"""Deterministic execution gate for workflow uncertainty decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from uncertainty.confidence_signal import OverallConfidence
from uncertainty.uncertainty_types import ConfidenceSource
from world.prediction_result import PredictionResult


class ExecutionDecision(str, Enum):
    """Typed decisions available at the workflow confidence branch."""

    CONTINUE = "continue"
    REFLECT = "reflect"
    RETRY_RETRIEVAL = "retry_retrieval"
    ASK_FOR_CLARIFICATION = "ask_for_clarification"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class UncertaintyGatePolicy:
    """Explicit thresholds controlling the deterministic execution gate."""

    reflect_uncertainty: float = 0.55
    retry_retrieval_uncertainty: float = 0.70
    clarification_uncertainty: float = 0.85
    clarification_confidence: float = 0.30
    deny_world_risk: float = 0.75
    max_retrieval_retries: int = 1

    def __post_init__(self) -> None:
        for name in (
            "reflect_uncertainty",
            "retry_retrieval_uncertainty",
            "clarification_uncertainty",
            "clarification_confidence",
            "deny_world_risk",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.max_retrieval_retries < 0:
            raise ValueError("max_retrieval_retries must be non-negative.")

    def to_dict(self) -> dict[str, float | int]:
        """Serialize configured thresholds for runtime diagnostics."""
        return {
            "reflect_uncertainty": self.reflect_uncertainty,
            "retry_retrieval_uncertainty": self.retry_retrieval_uncertainty,
            "clarification_uncertainty": self.clarification_uncertainty,
            "clarification_confidence": self.clarification_confidence,
            "deny_world_risk": self.deny_world_risk,
            "max_retrieval_retries": self.max_retrieval_retries,
        }


@dataclass(frozen=True, slots=True)
class ExecutionGateResult:
    """Decision, rule, and threshold evidence emitted by the gate."""

    decision: ExecutionDecision
    rule: str
    rationale: str
    thresholds: dict[str, float | int]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the gate result for diagnostics and artifacts."""
        return {
            "decision": self.decision.value,
            "rule": self.rule,
            "rationale": self.rationale,
            "thresholds": dict(self.thresholds),
        }


@dataclass(frozen=True, slots=True)
class UncertaintyGate:
    """Apply bounded, deterministic policy to uncertainty and world risk."""

    policy: UncertaintyGatePolicy = UncertaintyGatePolicy()

    def evaluate(
        self,
        uncertainty: OverallConfidence,
        world_prediction: PredictionResult,
        *,
        retrieval_retry_count: int = 0,
        clarification_required: bool = False,
    ) -> ExecutionGateResult:
        """Choose one execution decision using explicit threshold precedence."""
        thresholds = self.policy.to_dict()
        if world_prediction.risk_score >= self.policy.deny_world_risk:
            return ExecutionGateResult(
                ExecutionDecision.ABSTAIN,
                "high_risk_denial",
                "Symbolic predicted risk meets the configured denial threshold.",
                thresholds,
            )
        if clarification_required or (
            uncertainty.uncertainty >= self.policy.clarification_uncertainty
            and uncertainty.confidence <= self.policy.clarification_confidence
        ):
            return ExecutionGateResult(
                ExecutionDecision.ASK_FOR_CLARIFICATION,
                "clarification_threshold",
                "Uncertainty is too high for safe execution without clarification.",
                thresholds,
            )
        retrieval_uncertainty = max(
            (
                signal.uncertainty
                for signal in uncertainty.signals
                if signal.source is ConfidenceSource.RETRIEVAL
            ),
            default=0.0,
        )
        if (
            retrieval_uncertainty >= self.policy.retry_retrieval_uncertainty
            and retrieval_retry_count < self.policy.max_retrieval_retries
        ):
            return ExecutionGateResult(
                ExecutionDecision.RETRY_RETRIEVAL,
                "retrieval_retry_threshold",
                "Retrieval uncertainty permits one bounded evidence retry.",
                thresholds,
            )
        if uncertainty.uncertainty >= self.policy.reflect_uncertainty:
            return ExecutionGateResult(
                ExecutionDecision.REFLECT,
                "reflection_threshold",
                "Overall uncertainty meets the configured reflection threshold.",
                thresholds,
            )
        return ExecutionGateResult(
            ExecutionDecision.CONTINUE,
            "confidence_clearance",
            "Uncertainty and symbolic risk are below configured thresholds.",
            thresholds,
        )
