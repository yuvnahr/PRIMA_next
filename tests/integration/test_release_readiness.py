"""Phase 15 fault injection and architecture call-trace evidence."""

from __future__ import annotations

import asyncio

import pytest

from benchmarks.campaign.config import ProviderConfig
from benchmarks.campaign.provider_session import FakeProvider, SharedProviderSession
from llm.generation_config import FallbackPolicy, GenerationConfig
from llm.llm_client import LLMClient
from llm.llm_types import LLMRequest, LLMResponse
from llm.provider import Provider, ProviderError
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType
from runtime import (
    DiagnosticMode,
    ExecutionOptions,
    ExecutionOutcome,
    ExecutionProfile,
    PrimaRequest,
    PrimaRuntime,
    TaskKind,
)
from runtime.contracts import RuntimeComponent
from uncertainty import DecisionType, OverallConfidence, UncertaintyBand, UncertaintyEstimator


class _TimeoutProvider(Provider):
    name = "fake"
    capabilities = FakeProvider.capabilities

    def __init__(self) -> None:
        pass

    def send(self, request: LLMRequest) -> LLMResponse:
        raise ProviderError("provider request timeout")


class _DenyLimiter:
    def allow(self, cost: int = 1) -> bool:
        return False


class _ResponseClient:
    provider_name = "test"
    capabilities = {"structured_output": ["json_schema"], "token_accounting": True}

    def __init__(self, text: str) -> None:
        self.text = text

    def chat(self, **_kwargs: object) -> LLMResponse:
        return LLMResponse(self.text)


class _FailingStore(InMemoryMemoryRepository):
    def add(self, note: MemoryNote) -> MemoryNote:
        raise OSError("memory store unavailable")


class _LowConfidence(UncertaintyEstimator):
    def estimate_from_subsystem_outputs(self, **_kwargs: object) -> OverallConfidence:
        return OverallConfidence(
            confidence=0.40,
            uncertainty=0.65,
            confidence_interval=(0.40, 0.40),
            uncertainty_band=UncertaintyBand.HIGH,
            decision_probabilities=(),
            recommended_decision=DecisionType.REFLECT,
            signals=(),
        )


def test_provider_timeout_retries_are_bounded_and_rate_limit_fails_closed() -> None:
    session = SharedProviderSession(
        ProviderConfig(kind="fake", model="fixture", revision="1", context_window=1024, retries=1),
        max_active_requests=1,
    )
    session.provider.provider = _TimeoutProvider()
    generation = GenerationConfig(model="fixture", provider="fake", retries=1)
    with pytest.raises(ProviderError, match="failed after 2 attempt"):
        session.client.chat("hello", generation_config=generation)
    assert session.telemetry()["request_attempts"] == 2
    assert session.telemetry()["retries"] == 1
    assert session.telemetry()["timeouts"] == 2

    limited = LLMClient(provider_name="fake", provider=FakeProvider(), rate_limiter=_DenyLimiter())
    with pytest.raises(RuntimeError, match="Rate limit exceeded"):
        limited.chat("hello", generation_config=GenerationConfig(model="fixture", provider="fake"))


@pytest.mark.parametrize("payload", ["not-json", '{"answer":"Alpha","evidence":["E1"]'])
def test_malformed_and_truncated_structured_output_fail_without_fallback(tmp_path, payload: str) -> None:
    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("Alpha is the answer.", MemoryType.SEMANTIC, note_id="source-1"))
    response = asyncio.run(
        PrimaRuntime(
            llm_client=_ResponseClient(payload),
            generation_config=GenerationConfig(
                model="fixture",
                provider="test",
                fallback_policy=FallbackPolicy.FAIL,
            ),
            memory_repository=repository,
            log_path=tmp_path / "runtime.log",
        ).execute(
            PrimaRequest(
                task_kind=TaskKind.FACTUAL_QA,
                profile=ExecutionProfile.SIMPLE_RAG,
                input_text="What is the answer?",
                options=ExecutionOptions(reasoning_mode="single_pass"),
            )
        )
    )
    assert response.outcome is ExecutionOutcome.FAILED
    assert response.diagnostics.provider["fallback_used"] is False
    assert response.errors
    assert response.diagnostics.provider["fallback_reason"]


def test_memory_store_failure_is_a_typed_runtime_failure(tmp_path) -> None:
    response = asyncio.run(
        PrimaRuntime(memory_repository=_FailingStore(), log_path=tmp_path / "runtime.log").execute(
            PrimaRequest(
                task_kind=TaskKind.DOCUMENT_INGESTION,
                profile=ExecutionProfile.INGESTION_ONLY,
                input_text="Store this document.",
            )
        )
    )
    assert response.outcome is ExecutionOutcome.FAILED
    assert any("memory store unavailable" in error for error in response.errors)
    assert RuntimeComponent.MODEL_EXECUTOR in response.diagnostics.skipped_components


def test_full_profile_call_trace_and_short_route_exclusions(tmp_path) -> None:
    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("Alpha connects to Bridge.", MemoryType.SEMANTIC, note_id="alpha"))
    runtime = PrimaRuntime(
        llm_client=_ResponseClient("A generated release response."),
        memory_repository=repository,
        uncertainty_estimator=_LowConfidence(),
        log_path=tmp_path / "runtime.log",
    )
    full = asyncio.run(
        runtime.execute(
            PrimaRequest(
                task_kind=TaskKind.CONVERSATION,
                profile=ExecutionProfile.PRIMA_FULL,
                input_text="Please remember my important career goal; I am extremely excited about Alpha and Bridge.",
                diagnostic_mode=DiagnosticMode.DIAGNOSTIC,
            )
        )
    )
    required = {
        RuntimeComponent.INPUT_PARSER,
        RuntimeComponent.STATE_MANAGER,
        RuntimeComponent.AFFECT_ENGINE,
        RuntimeComponent.DENSE_RETRIEVAL,
        RuntimeComponent.GRAPH_TRAVERSAL,
        RuntimeComponent.PLANNER,
        RuntimeComponent.WORLD_MODEL,
        RuntimeComponent.UNCERTAINTY_ESTIMATOR,
        RuntimeComponent.REFLECTION,
        RuntimeComponent.MODEL_EXECUTOR,
        RuntimeComponent.OUTPUT_VALIDATOR,
        RuntimeComponent.STATE_COMMIT,
        RuntimeComponent.MEMORY_COMMIT,
        RuntimeComponent.MAINTENANCE_EVENTS,
    }
    assert full.outcome is ExecutionOutcome.ANSWERED
    assert required <= set(full.diagnostics.executed_components)
    assert full.output_text != "Please remember my important career goal; I am extremely excited about Alpha and Bridge."
    assert full.diagnostics.trace_event_count == len(full.diagnostics.trace_events)
    reranker = full.diagnostics.component_details[RuntimeComponent.RERANKER.value]
    assert reranker["requested_backend"]
    assert reranker["active_backend"]

    model_only = asyncio.run(
        runtime.execute(
            PrimaRequest(
                task_kind=TaskKind.CONVERSATION,
                profile=ExecutionProfile.MODEL_ONLY,
                input_text="Give a short release response.",
            )
        )
    )
    assert RuntimeComponent.DENSE_RETRIEVAL in model_only.diagnostics.skipped_components
