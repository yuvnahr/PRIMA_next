"""High-level PRIMA workflow facade and default controller adapters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any

from action import ActionContext, ActionExecutor, ExecutionPolicy
from action.execution_result import ActionExecutionStatus, ExecutionResult
from affect.affect_engine import DynamicAffectEngine
from events.event_bus import EventBus
from events.event_types import EventType
from events.maintenance_events import maintenance_event
from llm.generation_config import GenerationConfig
from llm.llm_client import LLMClient
from memory.maintenance.background_supervisor import BackgroundMaintenanceSupervisor
from memory.maintenance.maintenance_pipeline import MaintenancePipeline
from memory.maintenance.memory_importance import MemoryImportanceEngine
from memory.memory_index import MemoryIndex
from memory.memory_note import MemoryNote
from memory.memory_repository import MemoryRepository
from memory.memory_types import MemoryLevel, MemoryType
from memory.retrieval.context_compressor import ContextCompressor
from memory.retrieval.retrieval_controller import RetrievalController
from memory.retrieval.retrieval_request import RetrievalRequest
from planning import PlanningContext, TaskPlanner
from planning.plan import Plan
from reasoning.controller import ReasoningController
from reasoning.models import AnswerResult, ReasoningBudget, ReasoningMode, ReasoningRequest
from reasoning.reflection_advisor import ReflectionAdvisor, ReflectionEvent
from reflection.reasoning_reflection_adapter import ReasoningReflectionAdapter
from reflection.reflection_context import ReflectionContext
from reflection.reflection_engine import ReflectionEngine
from state.cognitive_state import CognitiveState
from state.state_manager import StateManager
from uncertainty import (
    ConfidenceSignal,
    ConfidenceSource,
    ExecutionDecision,
    OverallConfidence,
    UncertaintyEstimator,
    UncertaintyGate,
)
from workflow.answer_generation import AnswerGenerationController, GenerationOutcome
from workflow.controller_registry import ControllerRegistry
from workflow.correction_loop import CorrectionBudget, CorrectionLoop
from workflow.execution_context import ExecutionContext
from workflow.orchestration_engine import OrchestrationEngine, RetryPolicy
from workflow.output_validation import OutputValidationController
from workflow.task_router import TaskRouter
from workflow.workflow_events import WorkflowEventBus
from workflow.workflow_state import WorkflowPhase
from world import PredictionResult, SimulationContext, StateSimulator


@dataclass(slots=True)
class StateLoadController:
    """Load session state at the start of every workflow route."""

    state_manager: StateManager
    phase: WorkflowPhase = WorkflowPhase.STATE_LOAD

    async def execute(self, context: ExecutionContext) -> CognitiveState:
        """Load isolated state for the request session."""
        return self.state_manager.load(str(context.metadata["state_session_id"]))


@dataclass(slots=True)
class StateCommitController:
    """Commit successful workflow state with optimistic concurrency."""

    state_manager: StateManager
    phase: WorkflowPhase = WorkflowPhase.STATE_COMMIT

    async def execute(self, context: ExecutionContext) -> dict[str, Any]:
        """Persist state unless the shaped outcome failed or was cancelled."""
        output = context.output if isinstance(context.output, dict) else {}
        outcome = str(output.get("outcome", "failed"))
        if outcome in {"failed", "cancelled"}:
            return {"state": context.cognitive_state, "committed": False}
        state = self.state_manager.save(
            str(context.metadata["state_session_id"]), context.cognitive_state, context.cognitive_state.version
        )
        return {"state": state, "committed": True}


@dataclass(slots=True)
class AffectController:
    """Workflow controller for affect analysis."""

    affect_engine: DynamicAffectEngine
    phase: WorkflowPhase = WorkflowPhase.AFFECT

    async def execute(self, context: ExecutionContext) -> Any:
        """Run affect analysis through the workflow."""
        return self.affect_engine.process(context.user_input, cognitive_state=context.cognitive_state)


@dataclass(slots=True)
class MemoryRetrievalController:
    """Workflow controller for memory retrieval."""

    retrieval_controller: RetrievalController
    top_k: int = 5
    phase: WorkflowPhase = WorkflowPhase.MEMORY_RETRIEVAL

    async def execute(self, context: ExecutionContext) -> Any:
        """Run memory retrieval using workflow-owned context."""
        affect_priors = {}
        affect_signals = ()
        if context.affect_update is not None:
            affect_priors = dict(getattr(context.affect_update, "retrieval_priors", {}))
            affect_signals = tuple(getattr(context.affect_update, "reflection_signals", ()))
        request = RetrievalRequest(
            query=context.user_input,
            memory_types=(
                (MemoryType.PROCEDURAL,)
                if str(context.metadata.get("task_kind")) == "tool_request"
                else tuple(item for item in MemoryType if item is not MemoryType.PROCEDURAL)
            ),
            top_k=self.top_k,
            state_filter={
                "goal_state": context.cognitive_state.goal_state,
                "task_state": context.cognitive_state.task_state,
                "confidence_state": context.cognitive_state.confidence_state,
                "environment_state": context.cognitive_state.environment_state,
            },
            affective_context={"retrieval_priors": affect_priors, "reflection_signal_count": len(affect_signals)},
            profile=str(context.metadata.get("profile", "prima_full")),
            task_kind=str(context.metadata.get("task_kind", "conversation")),
            required_reranker_backend=context.metadata.get("required_reranker_backend"),
        )
        return self.retrieval_controller.retrieve(request)


@dataclass(frozen=True, slots=True)
class EvidenceAcquisitionResult:
    """Bounded reasoning output and its latest retrieval response."""

    answer_result: AnswerResult
    latest_retrieval: Any | None = None


@dataclass(slots=True)
class EvidenceAcquisitionController:
    """Run the existing bounded ReasoningController inside the workflow."""

    reasoning_controller: ReasoningController
    retrieval_controller: RetrievalController
    context_compressor: ContextCompressor = field(default_factory=ContextCompressor)
    phase: WorkflowPhase = WorkflowPhase.EVIDENCE_ACQUISITION

    async def execute(self, context: ExecutionContext) -> EvidenceAcquisitionResult:
        """Acquire evidence without generating the final answer."""

        latest_retrieval: Any | None = None

        def retrieve(query: str) -> Any:
            nonlocal latest_retrieval
            latest_retrieval = self.retrieval_controller.retrieve(
                RetrievalRequest(
                    query=query,
                    top_k=int(context.metadata.get("top_k", 5)),
                    memory_types=tuple(item for item in MemoryType if item is not MemoryType.PROCEDURAL),
                    profile=str(context.metadata.get("profile", "prima_full")),
                    task_kind=str(context.metadata.get("task_kind", "factual_qa")),
                    required_reranker_backend=context.metadata.get("required_reranker_backend"),
                )
            )
            return latest_retrieval

        active_query = str(context.metadata.get("retrieval_query_override", context.user_input))
        mode = ReasoningMode(str(context.metadata.get("reasoning_mode", ReasoningMode.ADAPTIVE.value)))
        budget = ReasoningBudget(
            max_hops=int(context.metadata.get("max_hops", 3)),
            max_retrieval_calls=int(context.metadata.get("max_retrieval_calls", 3)),
            max_context_tokens=int(context.metadata.get("max_context_tokens", 1600)),
        )
        result = self.reasoning_controller.answer(
            ReasoningRequest(
                question=active_query,
                session_id=str(context.metadata.get("session_id", "")),
                mode=mode,
                budget=budget,
            ),
            retrieve=retrieve,
            synthesize=lambda _question, _evidence: ("", False, (), {}),
        )
        compression_enabled = context.metadata.get("context_compression_enabled", True)
        if not isinstance(compression_enabled, bool):
            raise TypeError("context_compression_enabled must be a boolean.")
        compressor = replace(self.context_compressor, enabled=compression_enabled)
        compressed = compressor.compress(result.evidence_references, budget.max_context_tokens)
        context.compressed_context = compressed
        result = replace(
            result,
            evidence_references=compressed.evidence,
            answer_diagnostics={**result.answer_diagnostics, "context_compression": compressed.to_dict()},
        )
        return EvidenceAcquisitionResult(result, latest_retrieval)


@dataclass(slots=True)
class PlanningController:
    """Workflow controller for planning."""

    planner: TaskPlanner = field(default_factory=TaskPlanner)
    phase: WorkflowPhase = WorkflowPhase.PLANNING

    async def execute(self, context: ExecutionContext) -> Plan:
        """Create or revise a pure reasoning plan from workflow-routed context."""
        planning_context = PlanningContext.from_subsystem_outputs(
            objective=context.user_input,
            cognitive_state=context.cognitive_state,
            retrieval_response=context.retrieval_response,
            affect_update=context.affect_update,
            reflection_signals=tuple(context.metadata.get("reflection_signals", ())),
            metadata={"execution_id": context.execution_id},
        )
        replan_reason = context.metadata.pop("replan_reason", None)
        if isinstance(context.plan, Plan) and replan_reason:
            return self.planner.replan(planning_context, context.plan, str(replan_reason))
        return self.planner.create_plan(planning_context)


@dataclass(slots=True)
class WorldSimulationController:
    """Run the existing deterministic symbolic world model inside the workflow."""

    simulator: StateSimulator = field(default_factory=StateSimulator)
    phase: WorkflowPhase = WorkflowPhase.WORLD_SIMULATION

    async def execute(self, context: ExecutionContext) -> PredictionResult:
        """Predict the selected plan under current state and execution policy."""
        policy = context.metadata.get("execution_policy")
        if not isinstance(policy, ExecutionPolicy):
            policy = ExecutionPolicy()
        intent = getattr(context.plan, "execution_intent", None)
        constraints = ["Unsupported claims must not proceed when uncertainty is high."]
        if not policy.allow_external_actions:
            constraints.append("Must not execute external tools without explicit policy permission.")
        if policy.require_sandbox:
            constraints.append("External actions must use the configured sandbox.")
        simulation = SimulationContext.from_inputs(
            cognitive_state=context.cognitive_state,
            plan=context.plan,
            constraints=tuple(constraints),
            metadata={
                "intended_action": getattr(getattr(intent, "intent_type", None), "value", "unknown"),
                "execution_target": (
                    "tool" if bool(getattr(intent, "requires_external_tool", False)) else "llm"
                ),
                "policy": policy.to_dict(),
                "symbolic": True,
                "learned_prediction": False,
            },
        )
        return self.simulator.simulate(simulation)


@dataclass(slots=True)
class UncertaintyController:
    """Aggregate workflow subsystem signals into one uncertainty estimate."""

    estimator: UncertaintyEstimator = field(default_factory=UncertaintyEstimator)
    phase: WorkflowPhase = WorkflowPhase.UNCERTAINTY_ESTIMATION

    async def execute(self, context: ExecutionContext) -> OverallConfidence:
        """Estimate uncertainty from evidence, plan, affect, state, and policy."""
        return self.estimator.estimate_from_subsystem_outputs(
            retrieval_response=context.retrieval_response,
            reflection_result=context.reflection_result,
            affect_update=context.affect_update,
            plan=context.plan,
            extra_signals=self._extra_signals(context),
        )

    def _extra_signals(self, context: ExecutionContext) -> tuple[ConfidenceSignal, ...]:
        signals: list[ConfidenceSignal] = []
        state_confidence = context.cognitive_state.confidence.get("overall_confidence")
        if state_confidence is not None:
            value = max(0.0, min(1.0, float(state_confidence)))
            signals.append(ConfidenceSignal(ConfidenceSource.STATE, value, 1.0 - value))

        policy = context.metadata.get("execution_policy")
        if not isinstance(policy, ExecutionPolicy):
            policy = ExecutionPolicy()
        intent = getattr(context.plan, "execution_intent", None)
        requires_tool = bool(getattr(intent, "requires_external_tool", False))
        tool_name = getattr(intent, "tool_name", None)
        allowed = not requires_tool or (
            policy.allow_external_actions and tool_name is not None and tool_name in policy.allowed_tools
        )
        signals.append(
            ConfidenceSignal(
                ConfidenceSource.POLICY,
                0.95 if allowed else 0.05,
                0.05 if allowed else 0.95,
                metadata={"allowed": allowed, "requires_external_tool": requires_tool, "tool_name": tool_name},
            )
        )
        return tuple(signals)


@dataclass(slots=True)
class ExecutionDecisionController:
    """Apply the typed uncertainty gate before action or model execution."""

    gate: UncertaintyGate = field(default_factory=UncertaintyGate)
    phase: WorkflowPhase = WorkflowPhase.EXECUTION_DECISION

    async def execute(self, context: ExecutionContext) -> Any:
        """Return the deterministic gate decision for this execution."""
        if not isinstance(context.uncertainty, OverallConfidence):
            raise TypeError("Execution decision requires an OverallConfidence estimate.")
        if not isinstance(context.world_prediction, PredictionResult):
            raise TypeError("Execution decision requires a symbolic PredictionResult.")
        result = self.gate.evaluate(
            context.uncertainty,
            context.world_prediction,
            retrieval_retry_count=int(context.metadata.get("retrieval_retry_count", 0)),
            clarification_required=bool(context.metadata.get("clarification_required", False)),
        )
        history = context.metadata.setdefault("execution_decision_history", [])
        if isinstance(history, list):
            history.append(result.to_dict())
        return result


@dataclass(slots=True)
class ReflectionController:
    """Workflow controller for adaptive reflection."""

    reflection_engine: ReflectionEngine
    reflection_advisor: ReflectionAdvisor | None = None
    phase: WorkflowPhase = WorkflowPhase.REFLECTION

    async def execute(self, context: ExecutionContext) -> Any:
        """Run reflection using only workflow-routed subsystem outputs."""
        retrieval_confidence = getattr(context.retrieval_response, "confidence", None)
        retrieved_memories = tuple(getattr(context.retrieval_response, "results", ()))
        affect_signals = (
            tuple(getattr(context.affect_update, "reflection_signals", ())) if context.affect_update else ()
        )
        failure_metadata = {
            "reason": "routine workflow reflection checkpoint",
            "severity": 0.15,
        }
        confidence_value = float(getattr(retrieval_confidence, "confidence", 0.5) if retrieval_confidence else 0.5)
        audit_sample = self._audit_sample(context.user_input)
        if retrieval_confidence and confidence_value < 0.30:
            failure_metadata = {
                "reason": "low confidence retrieval with ambiguous memories",
                "severity": 0.65,
                "threshold": 0.30,
            }
        elif retrieval_confidence and confidence_value < 0.55 and audit_sample:
            failure_metadata = {
                "reason": "low confidence audit sample",
                "severity": 0.85,
                "threshold": 0.55,
                "audit_sample": True,
            }
        reflection_history = tuple(context.metadata.get("reflection_history", ()))
        reflection_context = ReflectionContext(
            query=context.user_input,
            retrieved_memories=retrieved_memories,
            retrieval_confidence=retrieval_confidence,
            affect_confidence=(
                float(getattr(getattr(context.affect_update, "profile", None), "confidence", 0.5))
                if context.affect_update is not None
                else None
            ),
            cognitive_state=context.cognitive_state,
            emotional_state=context.cognitive_state.emotional_state,
            reflection_history=reflection_history,
            failure_metadata=failure_metadata,
            affect_signals=affect_signals,
        )
        result = self.reflection_engine.evaluate(reflection_context)
        if not result.should_reflect or self.reflection_advisor is None:
            return result
        event = self._event(context)
        query = str(context.metadata.get("retrieval_query_override", context.user_input))
        adapter_method = getattr(self.reflection_advisor, "advice_for_result", None)
        if callable(adapter_method):
            advice = adapter_method(event, result, query=query, stop_reason=event.value)
        else:
            advice = self.reflection_advisor.advise(event, context, query=query, stop_reason=event.value)
        return replace(result, advice=advice)

    def _event(self, context: ExecutionContext) -> ReflectionEvent:
        stop_reason = str(getattr(context.reasoning_result, "stop_reason", ""))
        mapped = {
            "no_new_evidence": ReflectionEvent.NO_NEW_EVIDENCE,
            "duplicate_query": ReflectionEvent.DUPLICATE_QUERY,
            "contradictory": ReflectionEvent.CONTRADICTORY_EVIDENCE,
        }.get(stop_reason)
        if mapped is not None:
            return mapped
        constraints = tuple(getattr(context.world_prediction, "constraint_predictions", ()))
        if any(float(getattr(item, "violation_probability", 0.0)) >= 0.55 for item in constraints):
            return ReflectionEvent.PLAN_CONSTRAINT_VIOLATION
        return ReflectionEvent.LOW_CONFIDENCE

    def _audit_sample(self, text: str) -> bool:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 100 < 15


@dataclass(slots=True)
class ActionController:
    """Workflow controller for action preparation."""

    action_executor: ActionExecutor = field(default_factory=ActionExecutor)
    phase: WorkflowPhase = WorkflowPhase.ACTION

    async def execute(self, context: ExecutionContext) -> Any:
        """Execute the planned action through policy-gated action tooling."""
        policy = context.metadata.get("execution_policy")
        if not isinstance(policy, ExecutionPolicy):
            policy = ExecutionPolicy()
        action_context = ActionContext(
            plan=context.plan,
            cognitive_state=context.cognitive_state,
            world_prediction=context.world_prediction,
            uncertainty=context.uncertainty,
            policy=policy,
            metadata={"execution_id": context.execution_id},
        )
        result = await self.action_executor.execute(action_context)
        payload = result.to_dict()
        payload["execution_result"] = result
        payload["plan"] = (
            context.plan.to_dict() if (context.plan is not None and hasattr(context.plan, "to_dict")) else context.plan
        )
        return payload


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Typed document indexing result."""

    memory_id: str
    memory_type: MemoryType


@dataclass(slots=True)
class DocumentIngestionController:
    """Validate and index a document without answer generation."""

    repository: MemoryRepository
    phase: WorkflowPhase = WorkflowPhase.DOCUMENT_INGESTION

    async def execute(self, context: ExecutionContext) -> IngestionResult:
        """Index one semantic source or explicitly imported procedure."""

        text = context.user_input.strip()
        if not text:
            raise ValueError("Document text must not be empty.")
        document_metadata = dict(context.metadata.get("document_metadata", {}))
        explicit_type = str(document_metadata.pop("memory_type", MemoryType.SEMANTIC.value))
        if explicit_type not in {MemoryType.SEMANTIC.value, MemoryType.PROCEDURAL.value}:
            raise ValueError("Document imports may create only semantic or procedural memory.")
        memory_type = MemoryType(explicit_type)
        note = MemoryNote.create(
            content=text,
            memory_type=memory_type,
            memory_level=(
                MemoryLevel.VERIFIED_PROCEDURE
                if memory_type is MemoryType.PROCEDURAL
                else MemoryLevel.SEMANTIC_ABSTRACTION
            ),
            context={
                "source": "document_ingestion",
                "explicit_import": memory_type is MemoryType.PROCEDURAL,
                **document_metadata,
            },
        )
        return IngestionResult(self.repository.add(note).id, memory_type)


@dataclass(slots=True)
class MemoryCommitController:
    """Commit an admitted conversational turn inside the workflow lifecycle."""

    repository: MemoryRepository
    importance_engine: MemoryImportanceEngine
    phase: WorkflowPhase = WorkflowPhase.MEMORY_COMMIT

    async def execute(self, context: ExecutionContext) -> dict[str, Any]:
        """Apply outcome-aware admission and persist an episodic exchange."""

        output = context.output if isinstance(context.output, dict) else {}
        outcome = str(output.get("outcome", "failed"))
        if outcome in {"failed", "cancelled", "abstained"}:
            return {"notes": (), "admission": {"stored": False, "policy": "successful_answers_only", "reason": outcome}}
        if str(context.metadata.get("task_kind")) == "tool_request":
            return self._commit_procedure(context)

        decision = self.importance_engine.decide(
            context.user_input,
            affect_update=context.affect_update,
            reflection_result=context.reflection_result,
        )
        if not decision.stored:
            return {"notes": (), "admission": {**decision.to_log_record(), "policy": "importance_threshold"}}
        affect_update = context.affect_update
        user_note = MemoryNote.create(
            content=context.user_input,
            memory_type=MemoryType.EPISODIC,
            affective_state=affect_update.to_dict() if affect_update is not None else {},
            context={
                "source": "prima_runtime",
                "role": "user",
                "execution_id": context.execution_id,
                "source_session_id": str(context.metadata.get("session_id", "")),
                "source_turn_id": str(context.metadata.get("turn_id", "")),
            },
            state_snapshot=getattr(context.cognitive_state, "state_snapshot", None),
            salience_score=float(getattr(affect_update, "salience_score", 0.0) or 0.0),
        )
        assistant_note = MemoryNote.create(
            content=str(output.get("text", "")),
            memory_type=MemoryType.EPISODIC,
            context={
                "source": "prima_runtime",
                "role": "assistant",
                "execution_id": context.execution_id,
                "source_session_id": str(context.metadata.get("session_id", "")),
                "source_turn_id": str(context.metadata.get("turn_id", "")),
            },
        )
        notes = (self.repository.add(user_note), self.repository.add(assistant_note))
        return {
            "notes": notes,
            "admission": {**decision.to_log_record(), "policy": "importance_threshold", "record_count": 2},
        }

    def _commit_procedure(self, context: ExecutionContext) -> dict[str, Any]:
        action = context.action_result if isinstance(context.action_result, dict) else {}
        result = action.get("execution_result")
        if not isinstance(result, ExecutionResult) or result.status is not ActionExecutionStatus.SUCCESS:
            return {
                "notes": (),
                "admission": {
                    "stored": False,
                    "policy": "verified_successful_procedures_only",
                    "reason": "not_successful",
                },
            }
        tool_count = int(result.metadata.get("tool_invocation_count", 0))
        if tool_count < 1:
            return {
                "notes": (),
                "admission": {
                    "stored": False,
                    "policy": "verified_successful_procedures_only",
                    "reason": "no_tool_execution",
                },
            }
        tool_names = tuple(
            str(step.metadata.get("tool_name"))
            for step in result.steps
            if step.metadata.get("tool_name")
        )
        procedure_steps = tuple(
            f"{step.action_id}:{step.message}"
            for step in result.steps
            if step.status is ActionExecutionStatus.SUCCESS
        )
        note = MemoryNote.create(
            content=(
                f"Verified successful procedure for: {context.user_input}. "
                f"Steps: {'; '.join(procedure_steps) or 'verified tool execution'}."
            ),
            memory_type=MemoryType.PROCEDURAL,
            memory_level=MemoryLevel.VERIFIED_PROCEDURE,
            context={
                "source": "verified_tool_execution",
                "verified_success": True,
                "execution_id": context.execution_id,
                "plan_id": str(getattr(context.plan, "plan_id", "")),
                "tool_invocation_count": tool_count,
                "tool_names": list(tool_names),
            },
        )
        saved = self.repository.add(note)
        return {
            "notes": (saved,),
            "admission": {
                "stored": True,
                "policy": "verified_successful_procedures_only",
                "record_count": 1,
            },
        }


@dataclass(slots=True)
class MaintenanceEnqueueController:
    """Publish admitted memories to the bounded cold-path queue without waiting."""

    repository: MemoryRepository
    supervisor: BackgroundMaintenanceSupervisor
    phase: WorkflowPhase = WorkflowPhase.MAINTENANCE_ENQUEUE

    async def execute(self, context: ExecutionContext) -> dict[str, Any]:
        """Enqueue maintenance events and return immediately."""

        notes = [note for note in context.memory_notes_created if isinstance(note, MemoryNote)]
        ingestion_id = str(getattr(context.ingestion_result, "memory_id", ""))
        if ingestion_id:
            ingested = self.repository.get(ingestion_id)
            if ingested is not None:
                notes.append(ingested)
        queued_ids: list[str] = []
        for note in notes:
            admission = dict(context.memory_admission) if isinstance(context.memory_admission, dict) else {}
            event = maintenance_event(
                EventType.MEMORY_ADMITTED,
                memory_id=note.id,
                source="prima_workflow",
                execution_id=context.execution_id,
                payload={key: value for key, value in admission.items() if isinstance(value, (str, int, float, bool))},
            )
            await self.supervisor.event_bus.publish(event)
            if self.supervisor.enqueue(event):
                queued_ids.append(event.event_id)
        return {
            "schema_version": "1.0",
            "queued_count": len(queued_ids),
            "event_ids": queued_ids,
            "reason": (
                "queued"
                if queued_ids
                else "maintenance_disabled"
                if not self.supervisor.enabled
                else "queue_saturated_or_duplicate"
                if notes
                else "no_admitted_memory"
            ),
            **self.supervisor.diagnostics(),
        }


@dataclass(slots=True)
class OutputController:
    """Workflow controller for final output shaping."""

    phase: WorkflowPhase = WorkflowPhase.OUTPUT

    async def execute(self, context: ExecutionContext) -> dict[str, Any]:
        """Shape an existing typed subsystem result without generating content."""

        gate_result = context.execution_decision
        decision = getattr(gate_result, "decision", None)
        correction_terminal = str(context.metadata.get("correction_terminal_action", ""))
        if correction_terminal == "ask_user":
            return {
                "text": "Please clarify the request before I continue.",
                "outcome": GenerationOutcome.ABSTAINED.value,
                "correction_attempts": [attempt.to_dict() for attempt in context.correction_attempts],
            }
        if correction_terminal == "abstain":
            return {
                "text": "I cannot safely complete this request.",
                "outcome": GenerationOutcome.ABSTAINED.value,
                "correction_attempts": [attempt.to_dict() for attempt in context.correction_attempts],
            }
        if gate_result is not None and decision is ExecutionDecision.ASK_FOR_CLARIFICATION:
            return {
                "text": "Please clarify the request before I continue.",
                "outcome": GenerationOutcome.ABSTAINED.value,
                "execution_decision": gate_result.to_dict(),
            }
        if gate_result is not None and decision in {
            ExecutionDecision.ABSTAIN,
            ExecutionDecision.RETRY_RETRIEVAL,
        }:
            return {
                "text": "I cannot safely complete this request with the available evidence.",
                "outcome": GenerationOutcome.ABSTAINED.value,
                "execution_decision": gate_result.to_dict(),
            }

        if context.generation_result is not None:
            result = context.generation_result
            return {
                "text": result.text,
                "outcome": result.outcome.value,
                "generation": result.to_dict(),
            }
        if context.ingestion_result is not None:
            return {
                "text": None,
                "outcome": "ingested",
                "memory_id": context.ingestion_result.memory_id,
                "memory_type": context.ingestion_result.memory_type.value,
            }
        if str(context.metadata.get("task_kind")) == "emotion_classification" and context.affect_update is not None:
            profile = context.affect_update.profile
            return {
                "text": None,
                "outcome": "classified",
                "classification": {
                    "dominant_emotion": profile.dominant_emotion,
                    "confidence": profile.confidence,
                    "emotions": dict(profile.emotions),
                },
            }
        if context.action_result is None:
            raise RuntimeError("Output shaping requires a generation, ingestion, classification, or action result.")
        return {
            "text": None,
            "outcome": "actioned",
            "action": context.action_result,
            "execution_id": context.execution_id,
        }


class PrimaWorkflow:
    """Facade for executing the PRIMA cognitive control bus."""

    def __init__(
        self,
        registry: ControllerRegistry,
        router: TaskRouter | None = None,
        event_bus: WorkflowEventBus | None = None,
        retry_policy: RetryPolicy | None = None,
        correction_loop: CorrectionLoop | None = None,
    ) -> None:
        self.engine = OrchestrationEngine(
            registry=registry,
            router=router,
            event_bus=event_bus,
            retry_policy=retry_policy,
            correction_loop=correction_loop,
        )

    async def run(self, user_input: str, context: ExecutionContext | None = None) -> ExecutionContext:
        """Run one cognitive workflow execution."""
        execution_context = context or ExecutionContext(user_input=user_input)
        if context is not None:
            execution_context.user_input = user_input
        return await self.engine.execute(execution_context)

    async def cancel(self, context: ExecutionContext) -> None:
        """Request cancellation for a running context."""
        await self.engine.cancel(context)

    @classmethod
    def from_controllers(
        cls,
        affect_engine: DynamicAffectEngine,
        retrieval_controller: RetrievalController,
        reflection_engine: ReflectionEngine,
        reasoning_controller: ReasoningController | None = None,
        llm_client: LLMClient | None = None,
        memory_repository: MemoryRepository | None = None,
        state_manager: StateManager | None = None,
        state_simulator: StateSimulator | None = None,
        uncertainty_estimator: UncertaintyEstimator | None = None,
        uncertainty_gate: UncertaintyGate | None = None,
        action_executor: ActionExecutor | None = None,
        context_compressor: ContextCompressor | None = None,
        reflection_advisor: ReflectionAdvisor | None = None,
        correction_budget: CorrectionBudget | None = None,
        maintenance_supervisor: BackgroundMaintenanceSupervisor | None = None,
        event_bus: WorkflowEventBus | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> PrimaWorkflow:
        """Create a workflow with default controller adapters."""
        repository = memory_repository or retrieval_controller.repository
        if not isinstance(repository, MemoryIndex):
            controller_index = getattr(retrieval_controller, "memory_index", None)
            repository = controller_index if isinstance(controller_index, MemoryIndex) else MemoryIndex(repository)
        if state_manager is None:
            raise ValueError("PrimaWorkflow requires an injected StateManager.")
        if maintenance_supervisor is None:
            maintenance_bus = EventBus()
            maintenance_supervisor = BackgroundMaintenanceSupervisor(
                MaintenancePipeline(repository, maintenance_bus).handle,
                event_bus=maintenance_bus,
            )
        generation_client = llm_client or LLMClient()
        generation_config = GenerationConfig(
            model=str(getattr(getattr(generation_client, "settings", None), "default_model", "default")),
            provider=str(getattr(generation_client, "provider_name", "unknown")),
        )
        registry = ControllerRegistry(
            controllers={
                WorkflowPhase.STATE_LOAD: StateLoadController(state_manager),
                WorkflowPhase.AFFECT: AffectController(affect_engine),
                WorkflowPhase.MEMORY_RETRIEVAL: MemoryRetrievalController(retrieval_controller),
                WorkflowPhase.EVIDENCE_ACQUISITION: EvidenceAcquisitionController(
                    reasoning_controller or ReasoningController(),
                    retrieval_controller,
                    context_compressor or ContextCompressor(),
                ),
                WorkflowPhase.PLANNING: PlanningController(),
                WorkflowPhase.WORLD_SIMULATION: WorldSimulationController(state_simulator or StateSimulator()),
                WorkflowPhase.UNCERTAINTY_ESTIMATION: UncertaintyController(
                    uncertainty_estimator or UncertaintyEstimator()
                ),
                WorkflowPhase.EXECUTION_DECISION: ExecutionDecisionController(
                    uncertainty_gate or UncertaintyGate()
                ),
                WorkflowPhase.REFLECTION: ReflectionController(
                    reflection_engine,
                    reflection_advisor or ReasoningReflectionAdapter(reflection_engine),
                ),
                WorkflowPhase.ACTION: ActionController(action_executor or ActionExecutor()),
                WorkflowPhase.ANSWER_GENERATION: AnswerGenerationController(
                    generation_client,
                    default_generation_config=generation_config,
                ),
                WorkflowPhase.OUTPUT_VALIDATION: OutputValidationController(),
                WorkflowPhase.DOCUMENT_INGESTION: DocumentIngestionController(repository),
                WorkflowPhase.OUTPUT: OutputController(),
                WorkflowPhase.STATE_COMMIT: StateCommitController(state_manager),
                WorkflowPhase.MEMORY_COMMIT: MemoryCommitController(repository, MemoryImportanceEngine(repository)),
                WorkflowPhase.MAINTENANCE_ENQUEUE: MaintenanceEnqueueController(
                    repository, maintenance_supervisor
                ),
            }
        )
        return cls(
            registry=registry,
            event_bus=event_bus,
            retry_policy=retry_policy,
            correction_loop=CorrectionLoop(correction_budget or CorrectionBudget()),
        )
