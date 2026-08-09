"""Phase 05 tests for the workflow-owned world and uncertainty branch."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace

from action import ActionContext, ActionExecutor, ExecutionResult
from llm.llm_types import LLMResponse
from memory.memory_repository import InMemoryMemoryRepository
from runtime import ExecutionOutcome, ExecutionProfile, PrimaRequest, PrimaRuntime, TaskKind
from runtime.contracts import RuntimeComponent
from uncertainty import (
    ConfidenceSignal,
    ConfidenceSource,
    DecisionType,
    ExecutionDecision,
    OverallConfidence,
    UncertaintyBand,
    UncertaintyEstimator,
    UncertaintyGate,
    UncertaintyGatePolicy,
)
from world import PredictionResult, SimulationContext, StateSimulator


class _ModelClient:
    provider_name = "test"

    def chat(self, **_kwargs: object) -> LLMResponse:
        return LLMResponse("Generated after confidence clearance.")


class _FixedEstimator(UncertaintyEstimator):
    def __init__(self, estimate: OverallConfidence) -> None:
        super().__init__()
        self.fixed_estimate = estimate

    def estimate_from_subsystem_outputs(self, **_kwargs: object) -> OverallConfidence:
        return self.fixed_estimate


@dataclass(slots=True)
class _CapturingActionExecutor(ActionExecutor):
    contexts: list[ActionContext] = field(default_factory=list)

    async def execute(self, context: ActionContext) -> ExecutionResult:
        self.contexts.append(context)
        return await ActionExecutor.execute(self, context)


class _HighRiskSimulator(StateSimulator):
    def simulate(self, context: SimulationContext) -> PredictionResult:
        return replace(super().simulate(context), risk_score=0.90)


def _estimate(
    confidence: float,
    uncertainty: float,
    signals: tuple[ConfidenceSignal, ...] = (),
) -> OverallConfidence:
    return OverallConfidence(
        confidence=confidence,
        uncertainty=uncertainty,
        confidence_interval=(confidence, confidence),
        uncertainty_band=UncertaintyBand.HIGH if uncertainty >= 0.55 else UncertaintyBand.LOW,
        decision_probabilities=(),
        recommended_decision=DecisionType.REFLECT if uncertainty >= 0.55 else DecisionType.CONTINUE,
        signals=signals,
    )


def _request(**metadata: object) -> PrimaRequest:
    return PrimaRequest(
        task_kind=TaskKind.CONVERSATION,
        profile=ExecutionProfile.PRIMA_FULL,
        input_text="Explain the current plan.",
        metadata=dict(metadata),
    )


def _runtime(tmp_path, estimate: OverallConfidence, **kwargs: object) -> PrimaRuntime:
    return PrimaRuntime(
        llm_client=_ModelClient(),
        memory_repository=InMemoryMemoryRepository(),
        uncertainty_estimator=_FixedEstimator(estimate),
        log_path=tmp_path / "runtime.log",
        **kwargs,
    )


def test_high_confidence_executes_with_world_and_uncertainty_inputs(tmp_path) -> None:
    executor = _CapturingActionExecutor()
    runtime = _runtime(tmp_path, _estimate(0.90, 0.10), action_executor=executor)

    response = asyncio.run(runtime.execute(_request()))

    assert response.outcome is ExecutionOutcome.ANSWERED
    assert response.diagnostics.execution_decision == ExecutionDecision.CONTINUE.value
    assert response.diagnostics.decision_rule == "confidence_clearance"
    assert RuntimeComponent.WORLD_MODEL in response.diagnostics.executed_components
    assert RuntimeComponent.UNCERTAINTY_ESTIMATOR in response.diagnostics.executed_components
    assert RuntimeComponent.REFLECTION not in response.diagnostics.executed_components
    assert executor.contexts[0].world_prediction is not None
    assert executor.contexts[0].uncertainty is not None


def test_low_confidence_routes_through_reflection(tmp_path) -> None:
    runtime = _runtime(tmp_path, _estimate(0.40, 0.65))

    response = asyncio.run(runtime.execute(_request()))

    assert response.diagnostics.execution_decision == ExecutionDecision.REFLECT.value
    assert RuntimeComponent.REFLECTION in response.diagnostics.executed_components


def test_clarification_route_skips_action_and_generation(tmp_path) -> None:
    executor = _CapturingActionExecutor()
    runtime = _runtime(tmp_path, _estimate(0.20, 0.90), action_executor=executor)

    response = asyncio.run(runtime.execute(_request()))

    assert response.outcome is ExecutionOutcome.ABSTAINED
    assert response.diagnostics.execution_decision == ExecutionDecision.ASK_FOR_CLARIFICATION.value
    assert not executor.contexts
    assert RuntimeComponent.MODEL_EXECUTOR not in response.diagnostics.executed_components


def test_high_symbolic_risk_denies_execution(tmp_path) -> None:
    executor = _CapturingActionExecutor()
    runtime = _runtime(
        tmp_path,
        _estimate(0.90, 0.10),
        action_executor=executor,
        state_simulator=_HighRiskSimulator(),
    )

    response = asyncio.run(runtime.execute(_request()))

    assert response.outcome is ExecutionOutcome.ABSTAINED
    assert response.diagnostics.execution_decision == ExecutionDecision.ABSTAIN.value
    assert response.diagnostics.decision_rule == "high_risk_denial"
    assert not executor.contexts


def test_disabled_profile_records_reason_and_replacement_rule(tmp_path) -> None:
    runtime = _runtime(tmp_path, _estimate(0.90, 0.10))
    request = PrimaRequest(
        task_kind=TaskKind.CONVERSATION,
        profile=ExecutionProfile.MODEL_ONLY,
        input_text="Hello",
    )

    response = asyncio.run(runtime.execute(request))

    detail = response.diagnostics.component_details[RuntimeComponent.WORLD_MODEL.value]
    assert detail["status"] == "disabled"
    assert detail["reason"] == "disabled by execution profile 'model_only'"
    assert detail["replacement_decision_rule"] == "profile_direct_execution"


def test_retrieval_retry_reenters_the_bounded_full_route_once(tmp_path) -> None:
    retrieval = ConfidenceSignal(ConfidenceSource.RETRIEVAL, 0.20, 0.80)
    runtime = _runtime(tmp_path, _estimate(0.40, 0.54, (retrieval,)))

    response = asyncio.run(runtime.execute(_request()))

    assert response.outcome is ExecutionOutcome.ANSWERED
    assert response.output_data["retrieval_retry_count"] == 1
    assert [item["decision"] for item in response.diagnostics.decision_history] == [
        ExecutionDecision.RETRY_RETRIEVAL.value,
        ExecutionDecision.CONTINUE.value,
    ]


def test_gate_thresholds_are_deterministic_and_retry_is_bounded() -> None:
    policy = UncertaintyGatePolicy(reflect_uncertainty=0.55, retry_retrieval_uncertainty=0.70)
    gate = UncertaintyGate(policy)
    prediction = StateSimulator().simulate(SimulationContext.from_inputs())
    retrieval = ConfidenceSignal(ConfidenceSource.RETRIEVAL, 0.20, 0.80)
    estimate = _estimate(0.40, 0.54, (retrieval,))

    first = gate.evaluate(estimate, prediction, retrieval_retry_count=0)
    second = gate.evaluate(estimate, prediction, retrieval_retry_count=1)
    boundary = gate.evaluate(_estimate(0.45, 0.55), prediction)

    assert first.decision is ExecutionDecision.RETRY_RETRIEVAL
    assert second.decision is ExecutionDecision.CONTINUE
    assert boundary.decision is ExecutionDecision.REFLECT
    assert first.thresholds == second.thresholds == policy.to_dict()
