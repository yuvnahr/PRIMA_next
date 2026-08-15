"""End-to-end canonical runtime and compatibility-adapter tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from llm.llm_types import LLMResponse
from llm.provider import ProviderError
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType
from reasoning.models import AnswerResult
from runtime import (
    ExecutionOptions,
    ExecutionOutcome,
    ExecutionProfile,
    ExecutionStatus,
    PrimaRequest,
    PrimaRuntime,
    RuntimeResult,
    TaskKind,
)
from workflow.controller_registry import ControllerRegistry
from workflow.execution_context import ExecutionContext
from workflow.prima_workflow import OutputController, PrimaWorkflow
from workflow.task_router import TaskRoute
from workflow.workflow_state import WorkflowPhase


class FakeLLMClient:
    """Deterministic injected model client used by integration tests."""

    provider_name = "test"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> LLMResponse:
        self.calls.append(dict(kwargs))
        if kwargs.get("response_format"):
            return LLMResponse(
                json.dumps({"answer": "Alpha", "evidence": ["E1"], "insufficient_information": False})
            )
        return LLMResponse("A generated response.")


class FailingLLMClient(FakeLLMClient):
    """Provider stub that proves failures remain typed."""

    def chat(self, **kwargs: object) -> LLMResponse:
        self.calls.append(dict(kwargs))
        raise ProviderError("provider offline")


def _runtime(tmp_path, client=None, repository=None) -> PrimaRuntime:
    return PrimaRuntime(
        llm_client=client or FakeLLMClient(),
        memory_repository=repository or InMemoryMemoryRepository(),
        log_path=tmp_path / "runtime.log",
    )


def test_conversation_generates_output_instead_of_echoing_input(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    request = PrimaRequest(
        task_kind=TaskKind.CONVERSATION,
        profile=ExecutionProfile.MODEL_ONLY,
        input_text="Do not echo this input.",
    )

    response = asyncio.run(runtime.execute(request))

    assert response.status is ExecutionStatus.COMPLETED
    assert response.outcome is ExecutionOutcome.ANSWERED
    assert response.output_text == "A generated response."
    assert response.output_text != request.input_text


def test_output_controller_rejects_missing_typed_result() -> None:
    with pytest.raises(RuntimeError, match="requires a generation"):
        asyncio.run(OutputController().execute(ExecutionContext(user_input="never echo me")))


def test_factual_qa_acquires_evidence_and_generates_inside_workflow(tmp_path) -> None:
    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("Alpha identifies Bridge.", MemoryType.SEMANTIC))
    runtime = _runtime(tmp_path, repository=repository)
    request = PrimaRequest(
        task_kind=TaskKind.FACTUAL_QA,
        profile=ExecutionProfile.SIMPLE_RAG,
        input_text="What identifies Bridge?",
        options=ExecutionOptions(reasoning_mode="single_pass"),
    )

    response = asyncio.run(runtime.execute(request))
    completed = [event.phase for event in runtime.workflow.engine.event_bus.events if event.phase]

    assert response.outcome is ExecutionOutcome.ANSWERED
    assert response.output_text == "Alpha"
    assert response.evidence
    assert WorkflowPhase.EVIDENCE_ACQUISITION in completed
    assert WorkflowPhase.ANSWER_GENERATION in completed


def test_ingestion_uses_workflow_without_generation(tmp_path) -> None:
    client = FakeLLMClient()
    repository = InMemoryMemoryRepository()
    runtime = _runtime(tmp_path, client=client, repository=repository)

    response = asyncio.run(
        runtime.execute(
            PrimaRequest(
                task_kind=TaskKind.DOCUMENT_INGESTION,
                profile=ExecutionProfile.INGESTION_ONLY,
                input_text="Nate likes tea.",
            )
        )
    )
    completed = [event.phase for event in runtime.workflow.engine.event_bus.events if event.phase]

    assert response.outcome is ExecutionOutcome.INGESTED
    assert response.output_text is None
    assert repository.get(str(response.output_data["memory_id"])) is not None
    assert WorkflowPhase.DOCUMENT_INGESTION in completed
    assert WorkflowPhase.ANSWER_GENERATION not in completed
    assert client.calls == []


def test_provider_failure_and_abstention_are_typed(tmp_path) -> None:
    failed = asyncio.run(
        _runtime(tmp_path, client=FailingLLMClient()).execute(
            PrimaRequest(
                task_kind=TaskKind.CONVERSATION,
                profile=ExecutionProfile.MODEL_ONLY,
                input_text="Generate an answer.",
            )
        )
    )
    abstained = asyncio.run(
        _runtime(tmp_path).execute(
            PrimaRequest(
                task_kind=TaskKind.FACTUAL_QA,
                profile=ExecutionProfile.SIMPLE_RAG,
                input_text="What is unknown?",
            )
        )
    )

    assert failed.status is ExecutionStatus.FAILED
    assert failed.outcome is ExecutionOutcome.FAILED
    assert failed.errors == ("provider offline",)
    assert failed.output_data["memory_admission"] == {
        "stored": False,
        "policy": "successful_answers_only",
        "reason": "failed",
    }
    assert abstained.outcome is ExecutionOutcome.ABSTAINED
    assert abstained.output_text and not abstained.output_text.lstrip().startswith("{")
    assert abstained.output_data["memory_admission"]["policy"] == "qa_read_only"


def test_classification_and_cancellation_have_typed_outcomes(tmp_path) -> None:
    classified = asyncio.run(
        _runtime(tmp_path).execute(
            PrimaRequest(
                task_kind=TaskKind.EMOTION_CLASSIFICATION,
                profile=ExecutionProfile.AFFECT_ONLY,
                input_text="I feel joyful.",
            )
        )
    )

    class CancellingController:
        phase = WorkflowPhase.ANSWER_GENERATION

        async def execute(self, _context):
            raise asyncio.CancelledError

    class CancellationRouter:
        def route(self, _context):
            return TaskRoute((WorkflowPhase.ANSWER_GENERATION,))

    cancelled_runtime = PrimaRuntime(
        workflow=PrimaWorkflow(
            ControllerRegistry({WorkflowPhase.ANSWER_GENERATION: CancellingController()}),
            router=CancellationRouter(),
        ),
        llm_client=FakeLLMClient(),
        memory_repository=InMemoryMemoryRepository(),
        log_path=tmp_path / "cancelled.log",
    )
    cancelled = asyncio.run(
        cancelled_runtime.execute(
            PrimaRequest(
                task_kind=TaskKind.CONVERSATION,
                profile=ExecutionProfile.MODEL_ONLY,
                input_text="Cancel this request.",
            )
        )
    )

    assert classified.outcome is ExecutionOutcome.CLASSIFIED
    assert classified.output_data["dominant_emotion"]
    assert cancelled.status is ExecutionStatus.CANCELLED
    assert cancelled.outcome is ExecutionOutcome.CANCELLED


def test_compatibility_methods_delegate_to_execute(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    original_execute = runtime.execute
    calls: list[TaskKind] = []

    async def tracked_execute(request: PrimaRequest):
        calls.append(request.task_kind)
        return await original_execute(request)

    runtime.execute = tracked_execute

    processed = runtime.process("Remember this generated turn.")
    processed_async = asyncio.run(runtime.process_async("Generate another turn."))
    answered = runtime.answer_question("What is unknown?")
    ingested = runtime.ingest_document("A document to index.")

    assert isinstance(processed, RuntimeResult)
    assert isinstance(processed_async, RuntimeResult)
    assert isinstance(answered, AnswerResult)
    assert isinstance(ingested, MemoryNote)
    assert calls == [
        TaskKind.CONVERSATION,
        TaskKind.CONVERSATION,
        TaskKind.FACTUAL_QA,
        TaskKind.DOCUMENT_INGESTION,
    ]
