from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from config.runtime_config import RuntimeConfig
from llm.generation_config import GenerationConfig
from llm.llm_types import LLMResponse
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType
from runtime import (
    ExecutionOptions,
    ExecutionOutcome,
    ExecutionProfile,
    PrimaRequest,
    PrimaRuntime,
    RuntimeComponent,
    TaskKind,
)
from workflow.execution_context import ExecutionContext
from workflow.task_router import TaskRouter


class _Client:
    provider_name = "test"
    capabilities = {"structured_output": ["json_schema"]}

    def chat(self, **kwargs) -> LLMResponse:
        if kwargs.get("response_schema"):
            return LLMResponse('{"answer":"Alpha","evidence":["E1"],"insufficient_information":false}')
        return LLMResponse("generated")


@pytest.mark.parametrize(
    "key",
    [
        "route", "task_kind", "profile", "execution_policy", "top_k", "max_hops",
        "max_retrieval_calls", "max_context_tokens", "context_compression_enabled",
        "max_llm_calls", "time_budget_seconds", "answer_reserve_seconds",
        "required_reranker_backend", "retrieval_query_override", "tool_results",
    ],
)
def test_request_metadata_cannot_override_execution_controls(key: str) -> None:
    with pytest.raises(ValidationError, match="reserved execution keys"):
        PrimaRequest(
            task_kind=TaskKind.CONVERSATION,
            profile=ExecutionProfile.MODEL_ONLY,
            input_text="hello",
            metadata={key: "untrusted"},
        )


def test_task_router_has_no_fallback() -> None:
    with pytest.raises(ValueError, match="trusted canonical route"):
        TaskRouter().route(ExecutionContext("unrouted"))


def test_retrieval_calls_results_and_budget_are_distinct(tmp_path) -> None:
    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("Alpha first evidence", MemoryType.SEMANTIC))
    repository.add(MemoryNote.create("Alpha second evidence", MemoryType.SEMANTIC))
    response = asyncio.run(
        PrimaRuntime(
            llm_client=_Client(),
            generation_config=GenerationConfig(model="fixture", provider="test"),
            memory_repository=repository,
            log_path=tmp_path / "runtime.log",
        ).execute(
            PrimaRequest(
                task_kind=TaskKind.FACTUAL_QA,
                profile=ExecutionProfile.SIMPLE_RAG,
                input_text="What is Alpha?",
                options=ExecutionOptions(
                    reasoning_mode="single_pass",
                    max_hops=3,
                    max_retrieval_calls=1,
                    max_context_tokens=128,
                ),
            )
        )
    )
    assert response.output_data["effective_budget"]["max_retrieval_calls"] == 1
    assert response.diagnostics.retrieval_count == response.diagnostics.retrieval_call_count == 1
    assert response.diagnostics.retrieval_result_count == len(response.evidence) >= 1
    assert response.diagnostics.accepted_correction_count == 0
    assert response.diagnostics.latency_ms > 0
    assert response.diagnostics.component_details[RuntimeComponent.RERANKER.value]["active_backend"]
    planned = set(response.diagnostics.planned_components)
    executed = set(response.diagnostics.executed_components)
    assert planned == executed | set(response.diagnostics.not_executed_components)
    assert executed <= set(response.diagnostics.enabled_components) <= planned


def test_disabled_reranker_is_not_reported_as_executed(tmp_path) -> None:
    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("Alpha evidence", MemoryType.SEMANTIC))
    response = asyncio.run(
        PrimaRuntime(
            llm_client=_Client(),
            generation_config=GenerationConfig(model="fixture", provider="test"),
            memory_repository=repository,
            runtime_config=RuntimeConfig(reranker_enabled=False, reranker_backend="disabled"),
            log_path=tmp_path / "runtime.log",
        ).execute(
            PrimaRequest(
                task_kind=TaskKind.FACTUAL_QA,
                profile=ExecutionProfile.SIMPLE_RAG,
                input_text="What is Alpha?",
                options=ExecutionOptions(reasoning_mode="single_pass"),
            )
        )
    )
    assert RuntimeComponent.RERANKER not in response.diagnostics.executed_components
    assert RuntimeComponent.RERANKER not in response.diagnostics.enabled_components
    assert response.diagnostics.component_details[RuntimeComponent.RERANKER.value]["status"] == "disabled"


def test_tool_request_has_typed_action_outcome_and_executor_diagnostics(tmp_path) -> None:
    response = asyncio.run(
        PrimaRuntime(
            llm_client=_Client(),
            memory_repository=InMemoryMemoryRepository(),
            log_path=tmp_path / "runtime.log",
        ).execute(
            PrimaRequest(
                task_kind=TaskKind.TOOL_REQUEST,
                profile=ExecutionProfile.PRIMA_FULL,
                input_text="Prepare a safe response action.",
            )
        )
    )
    assert response.outcome is ExecutionOutcome.ACTIONED
    assert "action" in response.output_data
    assert RuntimeComponent.TOOL_EXECUTOR in response.diagnostics.executed_components
    assert RuntimeComponent.MODEL_EXECUTOR in response.diagnostics.skipped_components


def test_fake_provider_state_events_admission_barrier_and_partial_tools(tmp_path):
    from dataclasses import replace

    from action import ActionExecutor, ExecutionPolicy
    from memory.maintenance.background_supervisor import MaintenanceBarrier, MaintenanceMode
    from state import JsonStateManager
    from tools import RegisteredTool, ToolExecutionStatus, ToolExecutor, ToolRegistry, ToolResult
    from workflow.workflow_events import WorkflowEventType
    from workflow.workflow_state import WorkflowPhase

    calls = []

    class Handler:
        async def execute(self, invocation):
            calls.append(invocation.invocation_id)
            if invocation.tool_name == "two":
                raise RuntimeError("fake second step failed")
            return ToolResult(invocation.invocation_id, "one", ToolExecutionStatus.SUCCESS, output="receipt")

    names = ("one", "two")
    tools = ToolRegistry(tuple(RegisteredTool(name, Handler(), cooperative_deadline=True) for name in names))
    repo = InMemoryMemoryRepository()
    path = tmp_path / "state.json"
    runtime = PrimaRuntime(
        llm_client=_Client(),
        memory_repository=repo,
        state_manager=JsonStateManager(path),
        action_executor=ActionExecutor(ToolExecutor(tools)),
        log_path=tmp_path / "runtime.log",
    )
    bus = runtime.workflow.engine.event_bus
    deliveries = []
    bus.subscribe(lambda event: deliveries.append((event.execution_id, event.event_type, event.phase)))

    async def run():
        await runtime.start_maintenance()
        try:
            answer = await runtime.execute(
                PrimaRequest(
                    task_kind=TaskKind.CONVERSATION,
                    profile=ExecutionProfile.MODEL_ONLY,
                    input_text="Hello.",
                    session_id="s",
                )
            )
            assert answer.output_text == "generated" and JsonStateManager(path).load("s").version == 1
            ingest = await runtime.execute(
                PrimaRequest(
                    task_kind=TaskKind.DOCUMENT_INGESTION,
                    profile=ExecutionProfile.INGESTION_ONLY,
                    input_text="Ada likes tea.",
                    session_id="s",
                )
            )
            assert ingest.outcome == ExecutionOutcome.INGESTED
            assert await runtime.apply_maintenance_barrier(
                MaintenanceMode.EVENTUAL, MaintenanceBarrier.BEFORE_FINALIZATION
            )
            assert runtime.maintenance_supervisor.diagnostics()["completed_count"] >= 1
            assert any(note.content == "Ada likes tea." for note in repo.list())
            previous = JsonStateManager(path).load("s").version
            registry = runtime.workflow.engine.registry
            planner = registry.get(WorkflowPhase.PLANNING)

            class FakePlan:
                phase = WorkflowPhase.PLANNING

                async def execute(self, context):
                    plan = await planner.execute(context)
                    return replace(
                        plan,
                        actions=tuple(
                            replace(plan.actions[0], action_id=name, metadata={"tool_name": name}) for name in names
                        ),
                    )

            registry.register(FakePlan())
            partial = await runtime.execute(
                PrimaRequest(
                    task_kind=TaskKind.TOOL_REQUEST,
                    profile=ExecutionProfile.PRIMA_FULL,
                    input_text="Prepare a safe response action.",
                    session_id="s",
                    options=ExecutionOptions(
                        execution_policy=ExecutionPolicy(allowed_tools=names, allow_external_actions=True)
                    ),
                )
            )
            assert partial.outcome == ExecutionOutcome.FAILED and len(set(calls)) == len(calls) == 2
            assert JsonStateManager(path).load("s").version == previous
            assert not repo.list(MemoryType.PROCEDURAL)
            assert deliveries == [(e.execution_id, e.event_type, e.phase) for e in bus.events]
            assert sum(kind == WorkflowEventType.WORKFLOW_STARTED for _, kind, _ in deliveries) == 3
        finally:
            await runtime.stop_maintenance()

    asyncio.run(run())
