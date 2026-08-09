from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from llm.llm_types import LLMResponse
from memory.memory_repository import InMemoryMemoryRepository
from runtime.contracts import (
    ExecutionProfile,
    ExecutionStatus,
    PrimaRequest,
    PrimaResponse,
    RuntimeComponent,
    TaskKind,
)
from runtime.prima_runtime import PrimaRuntime
from runtime.route_profiles import InvalidRouteError, route_matrix, select_route
from workflow.execution_context import ExecutionContext
from workflow.task_router import TaskRouter


class _ModelClient:
    provider_name = "test"

    def chat(self, **_kwargs) -> LLMResponse:
        return LLMResponse("Generated response.")


def test_contract_serialization_round_trip() -> None:
    request = PrimaRequest(
        request_id="request-1",
        task_kind=TaskKind.FACTUAL_QA,
        profile=ExecutionProfile.SIMPLE_RAG,
        input_text="What does Nate like?",
        metadata={"top_k": 5},
    )

    restored = PrimaRequest.from_json(request.to_json())
    assert restored == request
    assert request.to_dict()["schema_version"] == "1.0"
    assert request.to_dict()["task_kind"] == "factual_qa"

    response = PrimaResponse(
        request_id=request.request_id,
        task_kind=request.task_kind,
        profile=request.profile,
        status=ExecutionStatus.COMPLETED,
        output_text="tea",
    )
    assert PrimaResponse.from_dict(response.to_dict()) == response


def test_all_task_profile_pairs_have_deterministic_decisions() -> None:
    matrix = route_matrix()
    assert len(matrix) == len(TaskKind) * len(ExecutionProfile) == 25

    valid = 0
    for task_kind in TaskKind:
        for profile in ExecutionProfile:
            planned = matrix[(task_kind, profile)]
            if planned is None:
                with pytest.raises(InvalidRouteError):
                    select_route(task_kind, profile)
                continue
            valid += 1
            assert select_route(task_kind, profile) is planned
            assert len(planned.components) == len(set(planned.components))
            assert set(planned.components).isdisjoint(planned.skipped_components)
            assert set(planned.components) | set(planned.skipped_components) == set(RuntimeComponent)
    assert valid == 9


def test_every_valid_contract_route_has_a_workflow_phase_plan() -> None:
    plans = []
    for (task_kind, profile), route in route_matrix().items():
        if route is None:
            continue
        plan = TaskRouter().route(
            ExecutionContext(
                user_input="route",
                metadata={"task_kind": task_kind.value, "profile": profile.value},
            )
        )
        assert plan.phases
        assert len(plan.phases) == len(set(plan.phases))
        plans.append(plan)
    assert len(plans) == 9


def test_short_routes_exclude_irrelevant_qa_components() -> None:
    affect = select_route(TaskKind.EMOTION_CLASSIFICATION, ExecutionProfile.AFFECT_ONLY)
    ingestion = select_route(TaskKind.DOCUMENT_INGESTION, ExecutionProfile.INGESTION_ONLY)
    qa_components = {
        RuntimeComponent.QUERY_REWRITER,
        RuntimeComponent.DENSE_RETRIEVAL,
        RuntimeComponent.SPARSE_RETRIEVAL,
        RuntimeComponent.PLANNER,
        RuntimeComponent.WORLD_MODEL,
        RuntimeComponent.UNCERTAINTY_ESTIMATOR,
        RuntimeComponent.REFLECTION,
        RuntimeComponent.MODEL_EXECUTOR,
    }

    assert qa_components.isdisjoint(affect.components)
    assert qa_components.isdisjoint(ingestion.components)
    assert RuntimeComponent.EMOTION_CLASSIFIER in affect.components
    assert RuntimeComponent.DOCUMENT_ENCODER in ingestion.components


def test_invalid_request_and_route_configuration() -> None:
    with pytest.raises(ValidationError):
        PrimaRequest(
            task_kind=TaskKind.CONVERSATION,
            profile=ExecutionProfile.MODEL_ONLY,
            input_text="   ",
        )

    with pytest.raises(InvalidRouteError, match="ingestion_only.*conversation"):
        select_route(TaskKind.CONVERSATION, ExecutionProfile.INGESTION_ONLY)


def test_execute_reports_planned_executed_and_skipped_components(tmp_path) -> None:
    runtime = PrimaRuntime(
        llm_client=_ModelClient(),
        memory_repository=InMemoryMemoryRepository(),
        log_path=tmp_path / "runtime.log",
    )
    request = PrimaRequest(
        task_kind=TaskKind.CONVERSATION,
        profile=ExecutionProfile.MODEL_ONLY,
        input_text="Hello",
    )

    response = asyncio.run(runtime.execute(request))

    assert response.status is ExecutionStatus.COMPLETED
    assert response.diagnostics.planned_components
    assert RuntimeComponent.POLICY_ROUTER in response.diagnostics.executed_components
    assert RuntimeComponent.MODEL_EXECUTOR in response.diagnostics.executed_components
    assert RuntimeComponent.AFFECT_ENGINE in response.diagnostics.skipped_components
    assert RuntimeComponent.MODEL_EXECUTOR not in response.diagnostics.skipped_components
    assert set(response.diagnostics.executed_components) | set(response.diagnostics.skipped_components) == set(
        RuntimeComponent
    )
    assert not set(response.diagnostics.executed_components) - set(response.diagnostics.planned_components)


def test_ingestion_and_emotion_use_short_canonical_routes(tmp_path) -> None:
    runtime = PrimaRuntime(
        llm_client=_ModelClient(),
        memory_repository=InMemoryMemoryRepository(),
        log_path=tmp_path / "runtime.log",
    )
    ingestion = asyncio.run(
        runtime.execute(
            PrimaRequest(
                task_kind=TaskKind.DOCUMENT_INGESTION,
                profile=ExecutionProfile.INGESTION_ONLY,
                input_text="Nate likes tea.",
            )
        )
    )
    emotion = asyncio.run(
        runtime.execute(
            PrimaRequest(
                task_kind=TaskKind.EMOTION_CLASSIFICATION,
                profile=ExecutionProfile.AFFECT_ONLY,
                input_text="I feel joyful.",
            )
        )
    )

    assert ingestion.status is ExecutionStatus.COMPLETED
    assert ingestion.output_text is None
    assert RuntimeComponent.MODEL_EXECUTOR in ingestion.diagnostics.skipped_components
    assert emotion.status is ExecutionStatus.COMPLETED
    assert emotion.output_data["dominant_emotion"]
    assert RuntimeComponent.DENSE_RETRIEVAL in emotion.diagnostics.skipped_components


def test_sync_wrapper_rejects_an_active_event_loop(tmp_path) -> None:
    runtime = PrimaRuntime(
        llm_client=_ModelClient(),
        memory_repository=InMemoryMemoryRepository(),
        log_path=tmp_path / "runtime.log",
    )
    request = PrimaRequest(
        task_kind=TaskKind.CONVERSATION,
        profile=ExecutionProfile.MODEL_ONLY,
        input_text="Hello",
    )

    assert runtime.execute_sync(request).status is ExecutionStatus.COMPLETED

    async def call_sync_inside_loop() -> None:
        with pytest.raises(RuntimeError, match=r"await PrimaRuntime\.execute"):
            runtime.execute_sync(request)

    asyncio.run(call_sync_inside_loop())
