"""Integrated PRIMA-NEXT runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from action import ActionExecutor
from affect.affect_engine import DynamicAffectEngine
from config.runtime_mode import RuntimeMode
from config.settings import get_settings
from events.event_bus import EventBus
from llm.generation_config import GenerationConfig
from llm.llm_client import LLMClient
from memory.maintenance.background_supervisor import (
    BackgroundMaintenanceSupervisor,
    InMemoryMaintenanceFailureStore,
    JsonlMaintenanceFailureStore,
    MaintenanceBarrier,
    MaintenanceMode,
)
from memory.maintenance.importance_types import MemoryAdmissionDecision
from memory.maintenance.maintenance_pipeline import MaintenancePipeline
from memory.maintenance.memory_importance import MemoryImportanceEngine
from memory.memory_index import MemoryIndex
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository, MemoryRepository
from memory.memory_types import MemoryType
from memory.repository_factory import RepositorySelection, select_memory_repository
from memory.retrieval.retrieval_controller import RetrievalController
from reasoning.config import reasoning_mode as configured_reasoning_mode
from reasoning.controller import ReasoningController
from reasoning.models import (
    AnswerResult,
    EvidenceItem,
    ReasoningBudget,
    ReasoningMode,
    SufficiencyStatus,
)
from reasoning.reflection_advisor import ReflectionAdvisor
from reflection.reasoning_reflection_adapter import ReasoningReflectionAdapter
from reflection.reflection_engine import ReflectionEngine
from runtime.contracts import (
    ComponentCapability,
    DiagnosticMode,
    EvidenceReference,
    ExecutionOutcome,
    ExecutionProfile,
    ExecutionStatus,
    PrimaRequest,
    PrimaResponse,
    RuntimeComponent,
    RuntimeDiagnostics,
    StateDelta,
    TaskKind,
)
from runtime.route_profiles import InvalidRouteError, RoutePlan, select_route
from runtime.runtime_context import RuntimeContext
from runtime.runtime_metrics import RuntimeMetrics
from runtime.runtime_result import RuntimeResult
from security.redaction import redact
from state.state_manager import InMemoryStateManager, JsonStateManager, StateManager
from uncertainty import UncertaintyEstimator, UncertaintyGate, UncertaintyGatePolicy
from workflow.correction_loop import CorrectionBudget
from workflow.execution_context import ExecutionContext
from workflow.orchestration_engine import WorkflowExecutionError
from workflow.prima_workflow import PrimaWorkflow
from workflow.workflow_state import WorkflowPhase, WorkflowStatus
from world import StateSimulator


class PrimaRuntime:
    """Production-facing runtime wrapper around the workflow orchestration layer."""

    def __init__(
        self,
        workflow: PrimaWorkflow | None = None,
        memory_repository: MemoryRepository | None = None,
        affect_engine: DynamicAffectEngine | None = None,
        reflection_engine: ReflectionEngine | None = None,
        llm_client: LLMClient | None = None,
        log_path: str | Path = "logs/runtime.log",
        mode: RuntimeMode | str = RuntimeMode.TEST,
        memory_backend: str | None = None,
        memory_path: str | None = None,
        state_manager: StateManager | None = None,
        state_path: str | Path | None = None,
        state_simulator: StateSimulator | None = None,
        uncertainty_estimator: UncertaintyEstimator | None = None,
        uncertainty_policy: UncertaintyGatePolicy | None = None,
        action_executor: ActionExecutor | None = None,
        reflection_advisor: ReflectionAdvisor | None = None,
        correction_budget: CorrectionBudget | None = None,
        maintenance_supervisor: BackgroundMaintenanceSupervisor | None = None,
        maintenance_enabled: bool = True,
        maintenance_queue_size: int = 128,
        maintenance_max_retries: int = 2,
        maintenance_failure_path: str | Path | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self.log_path = Path(log_path)
        self.mode = RuntimeMode(mode)
        self.memory_repository, self.repository_selection = select_memory_repository(
            self.mode, memory_repository, backend=memory_backend, path=memory_path
        )
        self.memory_index = MemoryIndex.for_repository(self.memory_repository)
        self._owns_maintenance_supervisor = maintenance_supervisor is None
        self._maintenance_enabled = maintenance_enabled
        self._maintenance_queue_size = maintenance_queue_size
        self._maintenance_max_retries = maintenance_max_retries
        self._maintenance_failure_path = Path(maintenance_failure_path) if maintenance_failure_path else None
        self.maintenance_supervisor = maintenance_supervisor or self._create_maintenance_supervisor()
        if state_manager is not None:
            if self.mode is RuntimeMode.PRODUCTION and type(state_manager) is InMemoryStateManager:
                raise RuntimeError("Production mode requires configured persistent cognitive state storage.")
            self.state_manager = state_manager
        elif self.mode is RuntimeMode.PRODUCTION:
            if state_path is None:
                raise RuntimeError("Production preflight requires a configured state_path.")
            self.state_manager = JsonStateManager(state_path)
        else:
            self.state_manager = InMemoryStateManager()
        self.affect_engine = affect_engine or DynamicAffectEngine()
        self.reflection_engine = reflection_engine or ReflectionEngine()
        settings = get_settings()
        self.generation_config = generation_config or GenerationConfig(
            model=settings.default_model,
            provider=(llm_client.provider_name if llm_client is not None else settings.default_provider),
        )
        self.memory_importance_engine = MemoryImportanceEngine(self.memory_index)
        self.retrieval_controller = RetrievalController(
            self.memory_index,
            candidate_pool_size=int(os.getenv("PRIMA_RETRIEVAL_CANDIDATE_POOL_SIZE", "30")),
        )
        self.reflection_advisor = reflection_advisor or ReasoningReflectionAdapter(self.reflection_engine)
        self.correction_budget = correction_budget or CorrectionBudget()
        self.reasoning_controller = ReasoningController(reflection_advisor=self.reflection_advisor)
        self.llm_client = llm_client or LLMClient(provider_name=self.generation_config.provider, settings=settings)
        self.state_simulator = state_simulator or StateSimulator()
        self.uncertainty_estimator = uncertainty_estimator or UncertaintyEstimator()
        self.uncertainty_gate = UncertaintyGate(uncertainty_policy or UncertaintyGatePolicy())
        self.action_executor = action_executor or ActionExecutor()
        self.workflow = workflow or PrimaWorkflow.from_controllers(
            affect_engine=self.affect_engine,
            retrieval_controller=self.retrieval_controller,
            reflection_engine=self.reflection_engine,
            reasoning_controller=self.reasoning_controller,
            llm_client=self.llm_client,
            memory_repository=self.memory_index,
            state_manager=self.state_manager,
            state_simulator=self.state_simulator,
            uncertainty_estimator=self.uncertainty_estimator,
            uncertainty_gate=self.uncertainty_gate,
            action_executor=self.action_executor,
            reflection_advisor=self.reflection_advisor,
            correction_budget=self.correction_budget,
            maintenance_supervisor=self.maintenance_supervisor,
        )
        self.logger = self._build_logger(self.log_path)

    async def execute(self, request: PrimaRequest) -> PrimaResponse:
        """Execute one typed request through the workflow-owned lifecycle."""

        if not isinstance(request, PrimaRequest):
            raise TypeError("request must be a PrimaRequest")
        try:
            route = select_route(request.task_kind, request.profile)
        except InvalidRouteError as exc:
            return PrimaResponse(
                request_id=request.request_id,
                task_kind=request.task_kind,
                profile=request.profile,
                status=ExecutionStatus.REJECTED,
                outcome=ExecutionOutcome.FAILED,
                diagnostics=RuntimeDiagnosticsAssembler.invalid(
                    str(exc), self.mode, self.repository_selection, request.diagnostic_mode
                ),
                errors=(str(exc),),
            )
        started = time.perf_counter()
        execution_context = ExecutionContext(
            user_input=request.input_text,
            metadata={
                **request.metadata,
                "task_kind": request.task_kind.value,
                "profile": request.profile.value,
                "session_id": request.session_id or "",
                "state_session_id": request.session_id or request.request_id,
                "correction_budget": self.correction_budget.to_dict(),
                "generation_config": request.generation_config or self.generation_config,
            },
        )
        try:
            execution_context = await self.workflow.run(request.input_text, execution_context)
        except WorkflowExecutionError as exc:
            execution_context = exc.context
        execution_context.metadata["workflow_trace"] = [
            {
                "event_type": event.event_type.value,
                "phase": event.phase.value if event.phase is not None else None,
                "status": event.status.value,
                "timestamp": event.timestamp.isoformat(),
                "payload": dict(event.payload),
            }
            for event in self.workflow.engine.event_bus.events
            if event.execution_id == execution_context.execution_id
        ]
        return self._response_from_execution(
            request,
            route,
            execution_context,
            round((time.perf_counter() - started) * 1000, 3),
        )

    async def start_maintenance(self) -> None:
        """Start the local cold-path worker for an async application lifecycle."""

        await self.maintenance_supervisor.start()

    async def flush_maintenance(self) -> None:
        """Wait for a deterministic maintenance barrier."""

        await self.maintenance_supervisor.flush()

    async def stop_maintenance(self, *, graceful: bool = True) -> None:
        """Stop the local cold-path worker and optionally drain queued work."""

        await self.maintenance_supervisor.stop(graceful=graceful)

    async def apply_maintenance_barrier(
        self,
        mode: MaintenanceMode | str,
        barrier: MaintenanceBarrier | str,
    ) -> bool:
        """Apply a consistency barrier without importing benchmark concepts."""

        return await self.maintenance_supervisor.apply_barrier(
            MaintenanceMode(mode), MaintenanceBarrier(barrier)
        )

    def apply_maintenance_barrier_sync(
        self,
        mode: MaintenanceMode | str,
        barrier: MaintenanceBarrier | str,
    ) -> bool:
        """Apply a barrier and close its temporary sync-loop worker."""

        async def apply_and_stop() -> bool:
            flushed = await self.apply_maintenance_barrier(mode, barrier)
            await self.stop_maintenance(graceful=flushed)
            return flushed

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(apply_and_stop())
        raise RuntimeError("Use 'await PrimaRuntime.apply_maintenance_barrier(...)' inside an active event loop.")

    def execute_sync(self, request: PrimaRequest) -> PrimaResponse:
        """Run :meth:`execute` only when the caller does not own an event loop."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.execute(request))
        raise RuntimeError(
            "PrimaRuntime.execute_sync() cannot run inside an active event loop; "
            "use 'await PrimaRuntime.execute(request)' instead."
        )

    def reset(self, mode: str = "isolated", preserve_repository: bool = True) -> None:
        """Reset runtime conversation state without deleting long-term memory by default."""

        if mode not in {"isolated", "persistent"}:
            raise ValueError("Runtime reset mode must be 'isolated' or 'persistent'.")
        if mode == "isolated" and not preserve_repository:
            if self.mode is not RuntimeMode.TEST:
                raise RuntimeError("Only test mode may replace memory with an ephemeral repository.")
        self.state_manager.reset()
        if mode == "isolated" and not preserve_repository:
            self.memory_repository = InMemoryMemoryRepository()
            self.memory_index = MemoryIndex.for_repository(self.memory_repository)
            if self._owns_maintenance_supervisor:
                self.maintenance_supervisor = self._create_maintenance_supervisor()
            self.repository_selection = RepositorySelection("in_memory", False)
            self.memory_importance_engine = MemoryImportanceEngine(self.memory_index)
            self.retrieval_controller = RetrievalController(
                self.memory_index,
                candidate_pool_size=int(os.getenv("PRIMA_RETRIEVAL_CANDIDATE_POOL_SIZE", "30")),
            )
            self.reasoning_controller = ReasoningController(reflection_advisor=self.reflection_advisor)
            self.workflow = PrimaWorkflow.from_controllers(
                affect_engine=self.affect_engine,
                retrieval_controller=self.retrieval_controller,
                reflection_engine=self.reflection_engine,
                reasoning_controller=self.reasoning_controller,
                llm_client=self.llm_client,
                memory_repository=self.memory_index,
                state_manager=self.state_manager,
                state_simulator=self.state_simulator,
                uncertainty_estimator=self.uncertainty_estimator,
                uncertainty_gate=self.uncertainty_gate,
                action_executor=self.action_executor,
                reflection_advisor=self.reflection_advisor,
                correction_budget=self.correction_budget,
                maintenance_supervisor=self.maintenance_supervisor,
            )

    def _create_maintenance_supervisor(self) -> BackgroundMaintenanceSupervisor:
        event_bus = EventBus()
        pipeline = MaintenancePipeline(self.memory_index, event_bus)
        failure_store = (
            JsonlMaintenanceFailureStore(
                self._maintenance_failure_path
                or self.log_path.with_name("maintenance_failures.jsonl")
            )
            if self.mode is RuntimeMode.PRODUCTION
            else InMemoryMaintenanceFailureStore()
        )
        return BackgroundMaintenanceSupervisor(
            pipeline.handle,
            event_bus=event_bus,
            failure_store=failure_store,
            max_queue_size=self._maintenance_queue_size,
            max_retries=self._maintenance_max_retries,
            enabled=self._maintenance_enabled,
        )

    def answer_question(
        self,
        question: str,
        context: RuntimeContext | None = None,
        top_k: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        max_context_tokens: int | None = None,
        session_id: str | None = None,
        reasoning_mode: str | None = None,
        max_hops: int | None = None,
        diagnostics: bool = False,
    ) -> AnswerResult:
        """Compatibility adapter for factual QA through :meth:`execute`."""

        mode = ReasoningMode.DIAGNOSTIC if diagnostics else configured_reasoning_mode(reasoning_mode)
        response = self.execute_sync(
            PrimaRequest(
                task_kind=TaskKind.FACTUAL_QA,
                profile=ExecutionProfile.PRIMA_FULL,
                input_text=question,
                session_id=session_id or (context.session_id if context is not None else None),
                generation_config=self.generation_config.with_overrides(
                    provider=provider,
                    model=model,
                ),
                diagnostic_mode=DiagnosticMode.DIAGNOSTIC if diagnostics else DiagnosticMode.STANDARD,
                metadata={
                    "top_k": int(top_k) if top_k is not None else 5,
                    "max_context_tokens": (
                        int(max_context_tokens)
                        if max_context_tokens is not None
                        else 1600
                    ),
                    "max_hops": int(max_hops or 3),
                    "reasoning_mode": mode.value,
                },
            )
        )
        result = self._answer_result_from_response(response)
        if context is not None:
            context.retrieved_memories = result.retrieved_memories
            context.confidence_score = result.confidence
            context.cognitive_state = self.state_manager.load(context.session_id)
            context.emotional_state = context.cognitive_state.emotional_state
        return result

    def ingest_document(self, text: str, metadata: dict[str, Any] | None = None) -> MemoryNote:
        """Compatibility adapter for workflow-owned document ingestion."""

        response = self.execute_sync(
            PrimaRequest(
                task_kind=TaskKind.DOCUMENT_INGESTION,
                profile=ExecutionProfile.INGESTION_ONLY,
                input_text=text,
                metadata={"document_metadata": dict(metadata or {})},
            )
        )
        memory_id = str(response.output_data.get("memory_id", ""))
        note = self.memory_repository.get(memory_id) if memory_id else None
        if note is None:
            raise RuntimeError("Document ingestion did not commit a memory.")
        return note

    def runtime_manifest(self) -> dict[str, Any]:
        """Return schema-versioned runtime/storage configuration for benchmark manifests."""

        return {
            "schema_version": "1.0",
            "runtime_mode": self.mode.value,
            "memory_repository": self.repository_selection.backend,
            "memory_persistent": self.repository_selection.persistent,
            "memory_path": self.repository_selection.path,
            "state_repository": "json" if isinstance(self.state_manager, JsonStateManager) else "in_memory",
            "generation": self.generation_config.to_dict(),
            "provider_capabilities": dict(getattr(self.llm_client, "capabilities", {})),
            "maintenance": self.maintenance_supervisor.diagnostics(),
        }

    def process(self, user_input: str, context: RuntimeContext | None = None) -> RuntimeResult:
        """Compatibility adapter for a synchronous canonical conversation."""

        response = self.execute_sync(self._conversation_request(user_input, context))
        result = self._runtime_result_from_response(response)
        self._update_compatibility_context(context, result)
        return result

    async def process_async(self, user_input: str, context: RuntimeContext | None = None) -> RuntimeResult:
        """Compatibility adapter for an asynchronous canonical conversation."""

        response = await self.execute(self._conversation_request(user_input, context))
        result = self._runtime_result_from_response(response)
        self._update_compatibility_context(context, result)
        return result

    def _conversation_request(self, user_input: str, context: RuntimeContext | None) -> PrimaRequest:
        return PrimaRequest(
            task_kind=TaskKind.CONVERSATION,
            profile=ExecutionProfile.PRIMA_FULL,
            input_text=user_input,
            session_id=context.session_id if context is not None else None,
            generation_config=self.generation_config,
            metadata={"turn_id": context.turn_id if context is not None else ""},
        )

    def _response_from_execution(
        self,
        request: PrimaRequest,
        route: RoutePlan,
        context: ExecutionContext,
        latency_ms: float,
    ) -> PrimaResponse:
        output = context.output if isinstance(context.output, dict) else {}
        workflow_status = context.workflow_state.status
        if workflow_status is WorkflowStatus.CANCELLED:
            outcome = ExecutionOutcome.CANCELLED
            status = ExecutionStatus.CANCELLED
        elif workflow_status is WorkflowStatus.FAILED:
            outcome = ExecutionOutcome.FAILED
            status = ExecutionStatus.FAILED
        else:
            try:
                outcome = ExecutionOutcome(str(output.get("outcome", "failed")))
            except ValueError:
                outcome = ExecutionOutcome.FAILED
            status = ExecutionStatus.FAILED if outcome is ExecutionOutcome.FAILED else ExecutionStatus.COMPLETED

        reasoning = context.reasoning_result if isinstance(context.reasoning_result, AnswerResult) else None
        evidence = tuple(
            EvidenceReference(
                source_id=item.source_id,
                text=item.text,
                score=item.retrieval_score,
                metadata={"hop": item.hop, "query": item.query, "provenance": dict(item.provenance)},
            )
            for item in (reasoning.evidence_references if reasoning is not None else ())
        )
        affect_state = (
            context.affect_update.profile.to_dict()
            if context.affect_update is not None and hasattr(context.affect_update.profile, "to_dict")
            else {}
        )
        created_ids = [note.id for note in context.memory_notes_created if isinstance(note, MemoryNote)]
        generation = context.generation_result
        generation_errors = tuple(str(error) for error in getattr(generation, "errors", ()))
        errors = tuple(dict.fromkeys((*context.workflow_state.errors, *generation_errors)))
        ingestion_id = str(getattr(context.ingestion_result, "memory_id", ""))
        if ingestion_id:
            created_ids.append(ingestion_id)
        state_changes: dict[str, Any] = {}
        if created_ids:
            state_changes["memory_ids_added"] = created_ids
        if context.affect_update is not None:
            state_changes["emotional_state"] = context.affect_update.emotional_state.to_dict()
        state_changes["version"] = context.cognitive_state.version
        state_changes["committed"] = bool(context.metadata.get("state_committed", False))
        output_data: dict[str, Any] = {
            "latency_ms": latency_ms,
            "affect_state": affect_state,
            "memory_ids_created": created_ids,
            "memory_admission": (
                dict(context.memory_admission)
                if isinstance(context.memory_admission, dict)
                else _default_memory_admission(request.task_kind, outcome, bool(ingestion_id))
            ),
            "confidence": float(reasoning.confidence if reasoning is not None else self._confidence_score(context)),
            "reflection_triggered": bool(context.correction_attempts) or bool(
                getattr(context.reflection_result, "should_reflect", False)
            ),
            "reflection_reasons": [
                dict(reason) for reason in getattr(context.reflection_result, "trigger_reasons", ())
            ],
            "generation": generation.to_dict() if generation is not None else {},
            "world_prediction": (
                context.world_prediction.to_dict() if context.world_prediction is not None else {}
            ),
            "uncertainty": context.uncertainty.to_dict() if context.uncertainty is not None else {},
            "execution_decision": (
                context.execution_decision.to_dict() if context.execution_decision is not None else {}
            ),
            "execution_decision_history": list(context.metadata.get("execution_decision_history", ())),
            "retrieval_retry_count": int(context.metadata.get("retrieval_retry_count", 0)),
            "retrieval": _retrieval_audit(context.retrieval_response),
            "context_compression": (
                context.compressed_context.to_dict() if context.compressed_context is not None else {}
            ),
            "correction_attempts": [attempt.to_dict() for attempt in context.correction_attempts],
            "correction_count": sum(attempt.accepted for attempt in context.correction_attempts),
            "maintenance": self.maintenance_supervisor.diagnostics(),
            "trace_summary": list(reasoning.trace_summary) if reasoning is not None else [],
        }
        if reasoning is not None:
            output_data.update(
                {
                    "sufficiency_status": reasoning.status.value,
                    "hop_count": reasoning.hop_count,
                    "stop_reason": reasoning.stop_reason,
                    "effective_budget": reasoning.effective_budget.to_dict(),
                    "reasoning_errors": list(reasoning.errors),
                    "reasoning_reflection_attempts": list(
                        reasoning.answer_diagnostics.get("reflection_attempts", ())
                    ),
                }
            )
        if ingestion_id:
            output_data["memory_id"] = ingestion_id
            output_data["memory_type"] = getattr(
                getattr(context.ingestion_result, "memory_type", MemoryType.SEMANTIC),
                "value",
                MemoryType.SEMANTIC.value,
            )
        if isinstance(output.get("classification"), dict):
            output_data.update(output["classification"])
        return PrimaResponse(
            request_id=request.request_id,
            task_kind=request.task_kind,
            profile=request.profile,
            status=status,
            outcome=outcome,
            output_text=str(output["text"]) if output.get("text") is not None else None,
            output_data=output_data,
            state_delta=StateDelta(changes=state_changes),
            evidence=evidence,
            diagnostics=RuntimeDiagnosticsAssembler.assemble(
                route,
                context,
                self.mode,
                self.repository_selection,
                context.cognitive_state.version,
                self.maintenance_supervisor.diagnostics(),
                latency_ms,
                request.diagnostic_mode,
                request.redact_prompts,
            ),
            errors=errors,
        )

    def _answer_result_from_response(self, response: PrimaResponse) -> AnswerResult:
        if response.outcome is ExecutionOutcome.FAILED:
            sufficiency = SufficiencyStatus.ERROR
        elif response.outcome is ExecutionOutcome.ABSTAINED:
            sufficiency = SufficiencyStatus.UNANSWERABLE
        else:
            raw_status = str(response.output_data.get("sufficiency_status", SufficiencyStatus.SUFFICIENT.value))
            sufficiency = SufficiencyStatus(raw_status)
        items: list[EvidenceItem] = []
        for index, evidence in enumerate(response.evidence):
            note = self.memory_repository.get(evidence.source_id)
            retrieval_result = None
            if note is not None:
                from memory.retrieval.retrieval_result import RetrievalResult

                retrieval_result = RetrievalResult(note, float(evidence.score or 0.0))
            items.append(
                EvidenceItem(
                    evidence_id=f"compat_{index}",
                    text=evidence.text,
                    source_id=evidence.source_id,
                    source_type="memory",
                    retrieval_score=float(evidence.score or 0.0),
                    hop=int(evidence.metadata.get("hop", 0)),
                    query=str(evidence.metadata.get("query", "")),
                    provenance=dict(evidence.metadata.get("provenance", {})),
                    result=retrieval_result,
                )
            )
        budget_data = dict(response.output_data.get("effective_budget", {}))
        budget = ReasoningBudget(
            max_hops=int(budget_data.get("max_hops", 3)),
            max_retrieval_calls=int(budget_data.get("max_retrieval_calls", 3)),
            max_llm_calls=int(budget_data.get("max_llm_calls", 4)),
            max_documents=int(budget_data.get("max_documents", 12)),
            max_context_tokens=int(budget_data.get("max_context_tokens", 1600)),
        )
        return AnswerResult(
            answer=response.output_text or "",
            status=sufficiency,
            confidence=float(response.output_data.get("confidence", 0.0)),
            evidence_references=tuple(items),
            hop_count=int(response.output_data.get("hop_count", 0)),
            stop_reason=str(
                response.output_data.get("stop_reason", response.outcome.value if response.outcome else "failed")
            ),
            effective_budget=budget,
            errors=response.errors,
            answer_diagnostics={
                **dict(response.output_data),
                "runtime_diagnostics": response.diagnostics.model_dump(mode="json"),
            },
        )

    def _runtime_result_from_response(self, response: PrimaResponse) -> RuntimeResult:
        answer = self._answer_result_from_response(response)
        created_notes = tuple(
            note
            for memory_id in response.output_data.get("memory_ids_created", [])
            if (note := self.memory_repository.get(str(memory_id))) is not None
        )
        attempts = tuple(response.output_data.get("correction_attempts", ()))
        before_answer = str(attempts[0].get("before_answer", "")) if attempts else response.output_text or ""
        after_answer = str(attempts[-1].get("after_answer", "")) if attempts else response.output_text or ""
        return RuntimeResult(
            final_response=response.output_text or "",
            prediction_before_reflection=before_answer,
            prediction_after_reflection=after_answer,
            affect_state=dict(response.output_data.get("affect_state", {})),
            retrieved_memories=answer.retrieved_memories,
            memory_notes_created=created_notes,
            reflection_triggered=bool(response.output_data.get("reflection_triggered", False)),
            confidence_score=float(response.output_data.get("confidence", 0.0)),
            latency_ms=float(response.output_data.get("latency_ms", 0.0)),
            errors=response.errors,
            reflection_reasons=tuple(dict(reason) for reason in response.output_data.get("reflection_reasons", [])),
            reflection_before_confidence=float(attempts[0].get("before_confidence", 0.0)) if attempts else 0.0,
            reflection_after_confidence=float(attempts[-1].get("after_confidence", 0.0)) if attempts else 0.0,
            reflection_utility_score=max((float(item.get("utility", 0.0)) for item in attempts), default=0.0),
            correction_count=int(response.output_data.get("correction_count", 0)),
            memory_admission=dict(response.output_data.get("memory_admission", {})),
            answer_diagnostics=answer.answer_diagnostics,
        )

    def _update_compatibility_context(self, context: RuntimeContext | None, result: RuntimeResult) -> None:
        if context is None:
            return
        context.retrieved_memories = result.retrieved_memories
        context.confidence_score = result.confidence_score
        context.cognitive_state = self.state_manager.load(context.session_id)
        context.emotional_state = context.cognitive_state.emotional_state

    def metrics_from_result(
        self, result: RuntimeResult, execution_context: ExecutionContext | None = None
    ) -> RuntimeMetrics:
        """Create serializable metrics for a runtime result."""
        plan = execution_context.plan if execution_context is not None else None
        plan_status = getattr(getattr(plan, "status", None), "value", None)
        return RuntimeMetrics(
            latency_ms=result.latency_ms,
            retrieval_count=len(result.retrieved_memories),
            memory_creation_count=len(result.memory_notes_created),
            reflection_trigger_count=1 if result.reflection_triggered else 0,
            emotion_classification=str(result.affect_state.get("dominant_emotion", "neutral")),
            confidence_score=result.confidence_score,
            planning_success=plan is not None and plan_status not in {"failed", None},
        )

    def _persist_turn_memory(
        self,
        user_input: str,
        execution_context: ExecutionContext,
        runtime_context: RuntimeContext,
    ) -> tuple[tuple[MemoryNote, ...], MemoryAdmissionDecision]:
        admission_decision = self.memory_importance_engine.decide(
            user_input,
            affect_update=execution_context.affect_update,
            reflection_result=execution_context.reflection_result,
        )
        if not admission_decision.stored:
            return (), admission_decision
        affect_update = execution_context.affect_update
        note = MemoryNote.create(
            content=user_input,
            memory_type=MemoryType.EPISODIC,
            affective_state=(
                affect_update.to_dict() if (affect_update is not None and hasattr(affect_update, "to_dict")) else {}
            ),
            context={
                "source": "prima_runtime",
                "execution_id": execution_context.execution_id,
                "source_session_id": runtime_context.session_id,
                "source_turn_id": runtime_context.turn_id,
            },
            state_snapshot=getattr(execution_context.cognitive_state, "state_snapshot", None),
            salience_score=float(getattr(affect_update, "salience_score", 0.0) or 0.0),
            timestamp=runtime_context.timestamp,
        )
        return (self.memory_repository.add(note),), admission_decision

    def _build_result(
        self,
        execution_context: ExecutionContext,
        created_notes: tuple[MemoryNote, ...],
        latency_ms: float,
        errors: tuple[str, ...],
        admission_decision: MemoryAdmissionDecision | None = None,
    ) -> RuntimeResult:
        affect_update = execution_context.affect_update
        retrieval_response = execution_context.retrieval_response
        reflection_result = execution_context.reflection_result
        output = execution_context.output if isinstance(execution_context.output, dict) else {}
        confidence_score = self._confidence_score(execution_context)
        workflow_errors = tuple(str(error) for error in execution_context.workflow_state.errors)
        return RuntimeResult(
            final_response=str(output.get("text") or ""),
            prediction_before_reflection=(
                execution_context.correction_attempts[0].before_answer
                if execution_context.correction_attempts
                else str(output.get("text") or "")
            ),
            prediction_after_reflection=(
                execution_context.correction_attempts[-1].after_answer
                if execution_context.correction_attempts
                else str(output.get("text") or "")
            ),
            affect_state=(
                affect_update.profile.to_dict()
                if (
                    affect_update is not None
                    and hasattr(affect_update, "profile")
                    and hasattr(affect_update.profile, "to_dict")
                )
                else {}
            ),
            retrieved_memories=tuple(getattr(retrieval_response, "results", ())),
            memory_notes_created=created_notes,
            reflection_triggered=bool(getattr(reflection_result, "should_reflect", False)),
            confidence_score=confidence_score,
            latency_ms=latency_ms,
            errors=errors + workflow_errors,
            reflection_reasons=tuple(dict(reason) for reason in getattr(reflection_result, "trigger_reasons", ())),
            reflection_before_confidence=float(getattr(reflection_result, "before_confidence", 0.0) or 0.0),
            reflection_after_confidence=float(getattr(reflection_result, "after_confidence", 0.0) or 0.0),
            reflection_utility_score=float(getattr(reflection_result, "utility_score", 0.0) or 0.0),
            correction_count=sum(attempt.accepted for attempt in execution_context.correction_attempts),
            memory_admission=admission_decision.to_log_record() if admission_decision is not None else {},
        )

    def _confidence_score(self, execution_context: ExecutionContext) -> float:
        plan_evaluation = getattr(getattr(execution_context.plan, "evaluation", None), "confidence", None)
        if plan_evaluation is not None:
            return round(float(plan_evaluation), 6)
        retrieval_confidence = getattr(
            getattr(execution_context.retrieval_response, "confidence", None), "confidence", None
        )
        if retrieval_confidence is not None:
            return round(float(retrieval_confidence), 6)
        affect_confidence = getattr(getattr(execution_context.affect_update, "profile", None), "confidence", None)
        return round(float(affect_confidence or 0.0), 6)

    def _update_runtime_context(
        self,
        runtime_context: RuntimeContext,
        execution_context: ExecutionContext,
        result: RuntimeResult,
    ) -> None:
        runtime_context.cognitive_state = execution_context.cognitive_state
        if execution_context.affect_update is not None and hasattr(execution_context.affect_update, "emotional_state"):
            runtime_context.emotional_state = execution_context.affect_update.emotional_state
        runtime_context.retrieved_memories = result.retrieved_memories
        runtime_context.active_plan = execution_context.plan
        runtime_context.reflection_signals = tuple(getattr(execution_context.reflection_result, "signals", ()))
        runtime_context.confidence_score = result.confidence_score

    def _log_turn(self, user_input: str, result: RuntimeResult, metrics: RuntimeMetrics) -> None:
        payload: dict[str, Any] = {
            "input": user_input,
            "retrieval_count": metrics.retrieval_count,
            "memory_creation_count": metrics.memory_creation_count,
            "memory_admission": dict(result.memory_admission),
            "reflection_events": metrics.reflection_trigger_count,
            "confidence_score": metrics.confidence_score,
            "latency_ms": metrics.latency_ms,
            "errors": list(result.errors),
        }
        self.logger.info(json.dumps(payload, sort_keys=True))

    def _build_logger(self, log_path: Path) -> logging.Logger:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"prima_runtime.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        return logger


def _retrieval_audit(response: Any) -> dict[str, Any]:
    diagnostics = dict(getattr(response, "diagnostics", {}))
    return {
        key: diagnostics[key]
        for key in (
            "query_rewrite",
            "profile",
            "task_kind",
            "executed_strategies",
            "skipped_strategies",
            "graph_reasoning_invoked",
            "memory_index",
            "reranker",
        )
        if key in diagnostics
    }


_CANONICAL_AVAILABLE = {
    RuntimeComponent.INPUT_PARSER,
    RuntimeComponent.TASK_SCHEDULER,
    RuntimeComponent.STATE_MANAGER,
    RuntimeComponent.PERSISTENT_STATE,
    RuntimeComponent.POLICY_ROUTER,
    RuntimeComponent.EXECUTION_DISPATCHER,
    RuntimeComponent.AFFECT_ENGINE,
    RuntimeComponent.QUERY_REWRITER,
    RuntimeComponent.DENSE_RETRIEVAL,
    RuntimeComponent.SPARSE_RETRIEVAL,
    RuntimeComponent.TEMPORAL_RETRIEVAL,
    RuntimeComponent.GRAPH_TRAVERSAL,
    RuntimeComponent.GRAPH_REASONING,
    RuntimeComponent.FUSION,
    RuntimeComponent.RERANKER,
    RuntimeComponent.CONFIDENCE_ESTIMATOR,
    RuntimeComponent.CONTEXT_COMPRESSOR,
    RuntimeComponent.PLANNER,
    RuntimeComponent.WORLD_MODEL,
    RuntimeComponent.UNCERTAINTY_ESTIMATOR,
    RuntimeComponent.REFLECTION,
    RuntimeComponent.MODEL_EXECUTOR,
    RuntimeComponent.OUTPUT_VALIDATOR,
    RuntimeComponent.DOCUMENT_ENCODER,
    RuntimeComponent.EMOTION_CLASSIFIER,
    RuntimeComponent.OUTPUT_SHAPER,
    RuntimeComponent.STATE_COMMIT,
    RuntimeComponent.MEMORY_INDEX,
    RuntimeComponent.MEMORY_COMMIT,
    RuntimeComponent.MAINTENANCE_EVENTS,
}


def _executed_components(context: ExecutionContext) -> tuple[RuntimeComponent, ...]:
    executed = {
        RuntimeComponent.INPUT_PARSER,
        RuntimeComponent.TASK_SCHEDULER,
        RuntimeComponent.POLICY_ROUTER,
        RuntimeComponent.EXECUTION_DISPATCHER,
    }
    completed = set(context.workflow_state.completed_phases)
    if WorkflowPhase.STATE_LOAD in completed:
        executed.update((RuntimeComponent.STATE_MANAGER, RuntimeComponent.PERSISTENT_STATE))
    if WorkflowPhase.AFFECT in completed:
        executed.update((RuntimeComponent.EMOTION_CLASSIFIER, RuntimeComponent.AFFECT_ENGINE))
    if completed & {WorkflowPhase.EVIDENCE_ACQUISITION, WorkflowPhase.MEMORY_RETRIEVAL}:
        retrieval_diagnostics = dict(getattr(context.retrieval_response, "diagnostics", {}))
        strategies = set(retrieval_diagnostics.get("executed_strategies", ()))
        executed.update(
            (
                RuntimeComponent.QUERY_REWRITER,
                RuntimeComponent.MEMORY_INDEX,
                RuntimeComponent.FUSION,
                RuntimeComponent.RERANKER,
                RuntimeComponent.CONFIDENCE_ESTIMATOR,
            )
        )
        strategy_components = {
            "dense": RuntimeComponent.DENSE_RETRIEVAL,
            "sparse": RuntimeComponent.SPARSE_RETRIEVAL,
            "temporal": RuntimeComponent.TEMPORAL_RETRIEVAL,
            "graph": RuntimeComponent.GRAPH_TRAVERSAL,
        }
        executed.update(component for name, component in strategy_components.items() if name in strategies)
        if retrieval_diagnostics.get("graph_reasoning_invoked"):
            executed.add(RuntimeComponent.GRAPH_REASONING)
    if (
        WorkflowPhase.EVIDENCE_ACQUISITION in completed
        and context.compressed_context is not None
        and bool(context.compressed_context.enabled)
    ):
        executed.add(RuntimeComponent.CONTEXT_COMPRESSOR)
    if WorkflowPhase.PLANNING in completed:
        executed.add(RuntimeComponent.PLANNER)
    if WorkflowPhase.WORLD_SIMULATION in completed:
        executed.add(RuntimeComponent.WORLD_MODEL)
    if WorkflowPhase.UNCERTAINTY_ESTIMATION in completed:
        executed.add(RuntimeComponent.UNCERTAINTY_ESTIMATOR)
    if WorkflowPhase.REFLECTION in completed:
        executed.add(RuntimeComponent.REFLECTION)
    if WorkflowPhase.ANSWER_GENERATION in completed and bool(getattr(context.generation_result, "llm_called", False)):
        executed.add(RuntimeComponent.MODEL_EXECUTOR)
    if WorkflowPhase.OUTPUT_VALIDATION in completed:
        executed.add(RuntimeComponent.OUTPUT_VALIDATOR)
    if WorkflowPhase.DOCUMENT_INGESTION in completed:
        executed.update(
            (RuntimeComponent.DOCUMENT_ENCODER, RuntimeComponent.MEMORY_INDEX, RuntimeComponent.MEMORY_COMMIT)
        )
    if WorkflowPhase.OUTPUT in completed:
        executed.add(RuntimeComponent.OUTPUT_SHAPER)
    if WorkflowPhase.MEMORY_COMMIT in completed:
        executed.add(RuntimeComponent.MEMORY_COMMIT)
    if (
        WorkflowPhase.MAINTENANCE_ENQUEUE in completed
        and int(context.maintenance_result.get("queued_count", 0)) > 0
    ):
        executed.add(RuntimeComponent.MAINTENANCE_EVENTS)
    if WorkflowPhase.STATE_COMMIT in completed and bool(context.metadata.get("state_committed", False)):
        executed.add(RuntimeComponent.STATE_COMMIT)
    return tuple(component for component in RuntimeComponent if component in executed)


class RuntimeDiagnosticsAssembler:
    """Build the single diagnostics contract used by every task route."""

    @staticmethod
    def assemble(
        route: RoutePlan,
        context: ExecutionContext,
        mode: RuntimeMode,
        repository: RepositorySelection,
        state_version: int,
        maintenance: dict[str, Any],
        latency_ms: float,
        diagnostic_mode: DiagnosticMode,
        redact_prompts: bool,
        note: str | None = None,
    ) -> RuntimeDiagnostics:
        return _route_diagnostics(
            route,
            context,
            mode,
            repository,
            state_version,
            maintenance,
            latency_ms,
            diagnostic_mode,
            redact_prompts,
            note,
        )

    @staticmethod
    def invalid(
        reason: str,
        mode: RuntimeMode,
        repository: RepositorySelection,
        diagnostic_mode: DiagnosticMode,
    ) -> RuntimeDiagnostics:
        return _invalid_route_diagnostics(reason, mode, repository, diagnostic_mode)


def _route_diagnostics(
    route: RoutePlan,
    context: ExecutionContext,
    mode: RuntimeMode,
    repository: RepositorySelection,
    state_version: int,
    maintenance: dict[str, Any],
    latency_ms: float,
    diagnostic_mode: DiagnosticMode,
    redact_prompts: bool,
    note: str | None = None,
) -> RuntimeDiagnostics:
    executed = _executed_components(context)
    retrieval_diagnostics = dict(getattr(context.retrieval_response, "diagnostics", {}))
    index_capabilities = dict(retrieval_diagnostics.get("memory_index", {}))
    unavailable = (
        {RuntimeComponent.GRAPH_TRAVERSAL, RuntimeComponent.GRAPH_REASONING}
        if index_capabilities and not bool(index_capabilities.get("graph", False))
        else set()
    )
    capabilities = tuple(
        ComponentCapability(
            component=component,
            available=component in _CANONICAL_AVAILABLE and component not in unavailable,
            reason=(
                None
                if component in _CANONICAL_AVAILABLE and component not in unavailable
                else "memory-index capability unavailable"
                if component in unavailable
                else "canonical execution integration deferred"
                if component in route.components
                else "not selected by route"
            ),
        )
        for component in RuntimeComponent
    )
    gate_result = context.execution_decision
    full_profile = route.profile is ExecutionProfile.PRIMA_FULL
    replacement_rule = "uncertainty_gate" if full_profile else "profile_direct_execution"
    component_details = {
        component.value: (
            {
                "status": "executed" if component in executed else "enabled_not_executed",
                "reason": "selected by prima_full profile",
                "replacement_decision_rule": replacement_rule,
            }
            if full_profile
            else {
                "status": "disabled",
                "reason": f"disabled by execution profile '{route.profile.value}'",
                "replacement_decision_rule": replacement_rule,
            }
        )
        for component in (RuntimeComponent.WORLD_MODEL, RuntimeComponent.UNCERTAINTY_ESTIMATOR)
    }
    skipped_strategies = dict(retrieval_diagnostics.get("skipped_strategies", {}))
    for name, component in {
        "dense": RuntimeComponent.DENSE_RETRIEVAL,
        "sparse": RuntimeComponent.SPARSE_RETRIEVAL,
        "temporal": RuntimeComponent.TEMPORAL_RETRIEVAL,
        "graph": RuntimeComponent.GRAPH_TRAVERSAL,
    }.items():
        component_details[component.value] = {
            "status": "executed" if component in executed else "disabled",
            "reason": skipped_strategies.get(name, "selected and invoked" if component in executed else "not selected"),
        }
    component_details[RuntimeComponent.GRAPH_REASONING.value] = {
        "status": "executed" if RuntimeComponent.GRAPH_REASONING in executed else "disabled",
        "reason": (
            "centrality reasoning invoked by graph traversal"
            if RuntimeComponent.GRAPH_REASONING in executed
            else skipped_strategies.get("graph", "graph retrieval not selected")
        ),
    }
    component_details[RuntimeComponent.RERANKER.value] = dict(retrieval_diagnostics.get("reranker", {}))
    if context.compressed_context is not None:
        component_details[RuntimeComponent.CONTEXT_COMPRESSOR.value] = {
            **context.compressed_context.to_dict(),
            "status": "executed" if context.compressed_context.enabled else "disabled",
            "reason": "token budget applied" if context.compressed_context.enabled else "no-compression ablation",
        }
    else:
        component_details[RuntimeComponent.CONTEXT_COMPRESSOR.value] = {
            "status": "disabled",
            "reason": "route has no evidence context",
        }
    component_details[RuntimeComponent.MEMORY_INDEX.value] = index_capabilities
    component_details[RuntimeComponent.MAINTENANCE_EVENTS.value] = {
        **maintenance,
        "status": "executed" if RuntimeComponent.MAINTENANCE_EVENTS in executed else "disabled",
        "reason": (
            "admitted-memory event enqueued"
            if RuntimeComponent.MAINTENANCE_EVENTS in executed
            else str(context.maintenance_result.get("reason", "not selected by route"))
        ),
    }
    generation = context.generation_result
    trace = tuple(
        redact(event, redact_prompts=redact_prompts)
        for event in context.metadata.get("workflow_trace", ())
        if isinstance(event, dict)
    )
    retrieval_count = len(getattr(context.retrieval_response, "results", ()) or ())
    provider_details = {
        "name": str(getattr(generation, "provider", "")),
        "model": str(getattr(generation, "model", "")),
        "capabilities": dict(getattr(generation, "provider_capabilities", {}) or {}),
        "fallback_policy": str(getattr(generation, "fallback_policy", "")),
        "fallback_used": bool(getattr(generation, "fallback_used", False)),
        "fallback_reason": getattr(generation, "fallback_reason", None),
    }
    reflection_count = sum(
        event.get("event_type") == "phase_completed" and event.get("phase") == "reflection"
        for event in trace
    )
    return RuntimeDiagnostics(
        route_name=route.name,
        planned_components=route.components,
        executed_components=executed,
        skipped_components=tuple(component for component in RuntimeComponent if component not in executed),
        capabilities=capabilities,
        notes=(note,) if note else (),
        runtime_mode=mode,
        memory_repository=repository.backend,
        memory_persistent=repository.persistent,
        state_version=state_version,
        execution_decision=getattr(getattr(gate_result, "decision", None), "value", None),
        decision_rule=getattr(gate_result, "rule", replacement_rule),
        decision_thresholds=dict(getattr(gate_result, "thresholds", {})),
        decision_history=tuple(context.metadata.get("execution_decision_history", ())),
        component_details=component_details,
        correction_budget=dict(context.metadata.get("correction_budget", {})),
        correction_attempts=tuple(attempt.to_dict() for attempt in context.correction_attempts),
        maintenance=maintenance,
        diagnostic_mode=diagnostic_mode,
        latency_ms=latency_ms,
        retrieval_count=retrieval_count,
        reflection_count=reflection_count,
        model_call_count=int(context.metadata.get("model_call_count", 0)),
        model_usage=dict(context.metadata.get("model_usage", {})),
        provider=redact(provider_details, redact_prompts=redact_prompts),
        trace_event_count=len(trace),
        trace_events=trace if diagnostic_mode is DiagnosticMode.DIAGNOSTIC else (),
    )


def _invalid_route_diagnostics(
    reason: str,
    mode: RuntimeMode,
    repository: RepositorySelection,
    diagnostic_mode: DiagnosticMode,
) -> RuntimeDiagnostics:
    return RuntimeDiagnostics(
        route_name="invalid",
        skipped_components=tuple(RuntimeComponent),
        capabilities=tuple(
            ComponentCapability(component=component, available=False, reason="invalid task/profile combination")
            for component in RuntimeComponent
        ),
        notes=(reason,),
        runtime_mode=mode,
        memory_repository=repository.backend,
        memory_persistent=repository.persistent,
        diagnostic_mode=diagnostic_mode,
    )


def _default_memory_admission(task: TaskKind, outcome: ExecutionOutcome, ingested: bool) -> dict[str, Any]:
    if task is TaskKind.DOCUMENT_INGESTION:
        return {"stored": ingested, "policy": "semantic_source_record", "reason": outcome.value}
    if task is TaskKind.FACTUAL_QA:
        return {"stored": False, "policy": "qa_read_only", "reason": outcome.value}
    if task is TaskKind.EMOTION_CLASSIFICATION:
        return {"stored": False, "policy": "classification_read_only", "reason": outcome.value}
    return {"stored": False, "policy": "successful_answers_only", "reason": outcome.value}
