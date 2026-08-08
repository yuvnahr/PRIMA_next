from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from action.execution_result import ActionExecutionStatus, ExecutionResult
from llm.llm_types import LLMResponse
from memory.graph.graph_reasoning_engine import GraphReasoningEngine
from memory.graph.graph_repository import GraphRepository
from memory.maintenance.memory_importance import MemoryImportanceEngine
from memory.memory_index import MemoryIndex
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryLevel, MemoryType
from memory.retrieval.context_compressor import ContextCompressor
from memory.retrieval.dense_strategy import DenseRetrievalStrategy
from memory.retrieval.graph_strategy import GraphTraversalStrategy
from memory.retrieval.reranker import Reranker, RerankerBackend
from memory.retrieval.retrieval_controller import RetrievalController
from memory.retrieval.retrieval_request import RetrievalRequest
from memory.retrieval.sparse_strategy import SparseRetrievalStrategy
from memory.retrieval.temporal_strategy import TemporalRetrievalStrategy
from planning import PlanningContext, TaskPlanner
from planning.planning_types import ActionType
from reasoning.models import EvidenceItem
from runtime.contracts import ExecutionProfile, PrimaRequest, RuntimeComponent, TaskKind
from runtime.prima_runtime import PrimaRuntime
from state.cognitive_state import CognitiveState
from workflow.execution_context import ExecutionContext
from workflow.prima_workflow import DocumentIngestionController, MemoryCommitController


def _note(note_id: str, text: str, memory_type: MemoryType = MemoryType.SEMANTIC) -> MemoryNote:
    level = MemoryLevel.VERIFIED_PROCEDURE if memory_type is MemoryType.PROCEDURAL else MemoryLevel.SEMANTIC_ABSTRACTION
    return MemoryNote.create(
        text,
        memory_type=memory_type,
        memory_level=level,
        embedding=(1.0, 0.0),
        note_id=note_id,
        context={"explicit_import": True} if memory_type is MemoryType.PROCEDURAL else None,
    )


class _GraphReasoningSpy(GraphReasoningEngine):
    def __init__(self, repository: GraphRepository) -> None:
        super().__init__(repository)
        self.calls = 0

    def centrality_scores(self) -> dict[str, float]:
        self.calls += 1
        return super().centrality_scores()


class _ModelClient:
    provider_name = "test"

    def chat(self, **_kwargs: object) -> LLMResponse:
        return LLMResponse('{"answer":"Alpha","evidence":["E1"],"insufficient_information":false}')


def test_profiles_select_real_strategies_and_full_profile_invokes_graph_reasoning() -> None:
    index = MemoryIndex(InMemoryMemoryRepository())
    index.add(_note("alpha", "Alpha bridge procedure context"))
    index.add(_note("bridge", "Bridge procedure context reaches Beta"))
    assert index.graph_repository is not None
    spy = _GraphReasoningSpy(index.graph_repository)
    graph = GraphTraversalStrategy(index.graph_repository, reasoning_engine=spy)
    controller = RetrievalController(
        index,
        strategies=[DenseRetrievalStrategy(), SparseRetrievalStrategy(), TemporalRetrievalStrategy(), graph],
        reranker=Reranker(backend=RerankerBackend.LEXICAL_FALLBACK),
    )

    simple = controller.retrieve(
        RetrievalRequest("Alpha bridge", query_embedding=(1.0, 0.0), profile="simple_rag")
    )
    assert simple.diagnostics["executed_strategies"] == ["dense", "sparse"]
    assert simple.diagnostics["skipped_strategies"]["graph"].startswith("disabled by execution profile")
    assert spy.calls == 0

    full = controller.retrieve(
        RetrievalRequest("Alpha connected bridge", query_embedding=(1.0, 0.0), profile="prima_full")
    )
    assert full.diagnostics["executed_strategies"] == ["dense", "sparse", "temporal", "graph"]
    assert full.diagnostics["graph_reasoning_invoked"] is True
    assert spy.calls == 1
    assert full.diagnostics["query_rewrite"]["original_query"] == "Alpha connected bridge"


def test_disabled_graph_capability_and_reranker_backend_are_explicit() -> None:
    index = MemoryIndex(InMemoryMemoryRepository(), graph_repository=None)
    controller = RetrievalController(index, reranker=Reranker(backend="lexical_fallback"))
    response = controller.retrieve(
        RetrievalRequest("anything", query_embedding=(1.0, 0.0), profile="prima_full")
    )
    assert response.diagnostics["memory_index"]["graph"] is False
    assert response.diagnostics["skipped_strategies"]["graph"] == "capability unavailable"
    assert response.diagnostics["reranker"] == {
        "requested_backend": "lexical_fallback",
        "active_backend": "lexical_fallback",
        "model": None,
        "fallback": True,
    }

    def unavailable(_model: str) -> object:
        raise RuntimeError("weights absent")

    with pytest.raises(RuntimeError, match="failed preflight"):
        RetrievalController(
            InMemoryMemoryRepository(),
            reranker=Reranker(backend="cross_encoder", model_loader=unavailable),
        )


def test_runtime_diagnostics_report_actual_profile_capabilities(tmp_path: Path) -> None:
    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("Alpha bridge source", MemoryType.SEMANTIC, note_id="alpha"))
    repository.add(MemoryNote.create("Bridge source related to Alpha", MemoryType.SEMANTIC, note_id="bridge"))
    runtime = PrimaRuntime(
        memory_repository=repository,
        llm_client=_ModelClient(),
        log_path=tmp_path / "runtime.log",
    )
    simple = asyncio.run(
        runtime.execute(
            PrimaRequest(
                task_kind=TaskKind.FACTUAL_QA,
                profile=ExecutionProfile.SIMPLE_RAG,
                input_text="What is Alpha bridge?",
                metadata={"reasoning_mode": "single_pass"},
            )
        )
    )
    assert RuntimeComponent.DENSE_RETRIEVAL in simple.diagnostics.executed_components
    assert RuntimeComponent.GRAPH_TRAVERSAL not in simple.diagnostics.executed_components
    assert simple.diagnostics.component_details["graph_traversal"]["status"] == "disabled"

    full = asyncio.run(
        runtime.execute(
            PrimaRequest(
                task_kind=TaskKind.FACTUAL_QA,
                profile=ExecutionProfile.PRIMA_FULL,
                input_text="What is connected to Alpha bridge?",
                metadata={"reasoning_mode": "single_pass"},
            )
        )
    )
    assert RuntimeComponent.GRAPH_TRAVERSAL in full.diagnostics.executed_components
    assert RuntimeComponent.GRAPH_REASONING in full.diagnostics.executed_components
    assert full.diagnostics.component_details["reranker"]["active_backend"] == "lexical_fallback"
    assert full.diagnostics.component_details["context_compressor"]["schema_version"] == "1.0"


def test_context_compression_preserves_evidence_and_citation_mapping() -> None:
    evidence = (
        EvidenceItem("ev-1", "one two three", "source-1", "memory", 0.9, 0, "query"),
        EvidenceItem("ev-2", "four five six", "source-2", "memory", 0.8, 0, "query"),
    )
    compressed = ContextCompressor().compress(evidence, token_budget=4)
    assert [item.evidence_id for item in compressed.evidence] == ["ev-1", "ev-2"]
    assert compressed.evidence[1].text == "four"
    assert compressed.citation_map == {"ev-1": "source-1", "ev-2": "source-2"}
    assert compressed.compression_loss == pytest.approx(2 / 6)
    assert compressed.to_dict()["schema_version"] == "1.0"

    ablation = ContextCompressor(enabled=False).compress(evidence, token_budget=1)
    assert ablation.evidence == evidence
    assert ablation.compression_loss == 0.0


def test_procedural_memory_is_lifecycle_indexed_and_changes_tool_planning_only() -> None:
    index = MemoryIndex(InMemoryMemoryRepository())
    with pytest.raises(ValueError, match="requires verified"):
        index.add(
            MemoryNote.create(
                "Unverified procedure",
                MemoryType.PROCEDURAL,
                MemoryLevel.VERIFIED_PROCEDURE,
                embedding=(1.0, 0.0),
            )
        )
    procedure = index.add(_note("proc-1", "Deploy service with verified rollback", MemoryType.PROCEDURAL))
    assert index.graph_repository is not None
    assert index.graph_repository.find_by_memory_id(procedure.id) is not None
    controller = RetrievalController(index, reranker=Reranker(backend="lexical_fallback"))

    factual = controller.retrieve(
        RetrievalRequest(
            "Deploy service rollback",
            query_embedding=(1.0, 0.0),
            profile="prima_full",
            task_kind="factual_qa",
        )
    )
    assert not factual.results

    tool = controller.retrieve(
        RetrievalRequest(
            "Deploy service rollback",
            query_embedding=(1.0, 0.0),
            profile="prima_full",
            task_kind="tool_request",
            memory_types=(MemoryType.PROCEDURAL,),
        )
    )
    plan = TaskPlanner().create_plan(
        PlanningContext.from_subsystem_outputs("Deploy service", retrieval_response=tool)
    )
    integration = next(action for action in plan.actions if action.action_type is ActionType.INTEGRATE_MEMORY)
    assert integration.metadata["procedural_memory_ids"] == ["proc-1"]


def test_only_verified_successful_tool_execution_produces_procedural_memory() -> None:
    index = MemoryIndex(InMemoryMemoryRepository())
    controller = MemoryCommitController(index, MemoryImportanceEngine(index))
    context = ExecutionContext(
        "Deploy service",
        cognitive_state=CognitiveState(),
        metadata={"task_kind": "tool_request"},
    )
    context.action_result = {
        "execution_result": ExecutionResult(
            status=ActionExecutionStatus.SUCCESS,
            intent_type="tool",
            metadata={"tool_invocation_count": 1},
        )
    }
    context.output = {"outcome": "answered"}
    stored = asyncio.run(controller.execute(context))
    assert stored["admission"]["stored"] is True
    assert index.list(MemoryType.PROCEDURAL)[0].context["verified_success"] is True

    failed_context = ExecutionContext("Bad deploy", metadata={"task_kind": "tool_request"})
    failed_context.action_result = {
        "execution_result": ExecutionResult(
            status=ActionExecutionStatus.FAILED,
            intent_type="tool",
            metadata={"tool_invocation_count": 1},
        )
    }
    failed_context.output = {"outcome": "answered"}
    rejected = asyncio.run(controller.execute(failed_context))
    assert rejected["admission"]["stored"] is False
    assert len(index.list(MemoryType.PROCEDURAL)) == 1


def test_explicit_procedure_import_uses_the_same_index_lifecycle() -> None:
    index = MemoryIndex(InMemoryMemoryRepository())
    context = ExecutionContext(
        "Restart the service, verify health, then roll back on failure.",
        metadata={"document_metadata": {"memory_type": "procedural", "source_name": "runbook"}},
    )
    result = asyncio.run(DocumentIngestionController(index).execute(context))
    note = index.get(result.memory_id)
    assert result.memory_type is MemoryType.PROCEDURAL
    assert note is not None and note.context["explicit_import"] is True
    assert index.graph_repository is not None
    assert index.graph_repository.find_by_memory_id(result.memory_id) is not None
