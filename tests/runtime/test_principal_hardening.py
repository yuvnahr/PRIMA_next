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
