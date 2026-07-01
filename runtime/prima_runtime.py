"""Integrated PRIMA-NEXT runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from affect.affect_engine import DynamicAffectEngine
from memory.maintenance.memory_importance import MemoryImportanceEngine
from memory.maintenance.importance_types import MemoryAdmissionDecision
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository, MemoryRepository
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_controller import RetrievalController
from reflection.reflection_engine import ReflectionEngine
from runtime.runtime_context import RuntimeContext
from runtime.runtime_metrics import RuntimeMetrics
from runtime.runtime_result import RuntimeResult
from workflow.execution_context import ExecutionContext
from workflow.prima_workflow import PrimaWorkflow


class PrimaRuntime:
    """Production-facing runtime wrapper around the workflow orchestration layer."""

    def __init__(
        self,
        workflow: PrimaWorkflow | None = None,
        memory_repository: MemoryRepository | None = None,
        affect_engine: DynamicAffectEngine | None = None,
        reflection_engine: ReflectionEngine | None = None,
        log_path: str | Path = "logs/runtime.log",
    ) -> None:
        self.memory_repository = memory_repository or InMemoryMemoryRepository()
        self.affect_engine = affect_engine or DynamicAffectEngine()
        self.reflection_engine = reflection_engine or ReflectionEngine()
        self.memory_importance_engine = MemoryImportanceEngine(self.memory_repository)
        self.workflow = workflow or PrimaWorkflow.from_controllers(
            affect_engine=self.affect_engine,
            retrieval_controller=RetrievalController(self.memory_repository),
            reflection_engine=self.reflection_engine,
        )
        self.log_path = Path(log_path)
        self.logger = self._build_logger(self.log_path)

    def process(self, user_input: str, context: RuntimeContext | None = None) -> RuntimeResult:
        """Process one user input through the integrated workflow."""
        return asyncio.run(self.process_async(user_input=user_input, context=context))

    async def process_async(self, user_input: str, context: RuntimeContext | None = None) -> RuntimeResult:
        """Async runtime entry point for callers that already own an event loop."""
        start = time.perf_counter()
        runtime_context = context or RuntimeContext()
        errors: list[str] = []
        execution_context = ExecutionContext(
            user_input=user_input,
            cognitive_state=runtime_context.cognitive_state,
            metadata={"session_id": runtime_context.session_id, "turn_id": runtime_context.turn_id},
        )

        try:
            execution_context = await self.workflow.run(user_input, execution_context)
            created_notes, admission_decision = self._persist_turn_memory(user_input, execution_context)
        except Exception as exc:
            errors.append(str(exc))
            execution_context.workflow_state.errors.append(str(exc))
            created_notes = ()
            admission_decision = None

        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        result = self._build_result(execution_context, created_notes, latency_ms, tuple(errors), admission_decision)
        metrics = self.metrics_from_result(result, execution_context)
        self._update_runtime_context(runtime_context, execution_context, result)
        self._log_turn(user_input, result, metrics)
        return result

    def metrics_from_result(self, result: RuntimeResult, execution_context: ExecutionContext | None = None) -> RuntimeMetrics:
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
            affective_state=(affect_update.to_dict() if (affect_update is not None and hasattr(affect_update, "to_dict")) else {}),
            context={"source": "prima_runtime", "execution_id": execution_context.execution_id},
            state_snapshot=getattr(execution_context.cognitive_state, "state_snapshot", None),
            salience_score=float(getattr(affect_update, "salience_score", 0.0) or 0.0),
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
            final_response=str(output.get("text") or execution_context.user_input),
            prediction_before_reflection=str(output.get("text") or execution_context.user_input),
            prediction_after_reflection=str(output.get("text") or execution_context.user_input),
            affect_state=(affect_update.profile.to_dict() if (affect_update is not None and hasattr(affect_update, "profile") and hasattr(affect_update.profile, "to_dict")) else {}),
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
            correction_count=1 if bool(getattr(reflection_result, "correction_applied", False)) else 0,
            memory_admission=admission_decision.to_log_record() if admission_decision is not None else {},
        )

    def _confidence_score(self, execution_context: ExecutionContext) -> float:
        plan_evaluation = getattr(getattr(execution_context.plan, "evaluation", None), "confidence", None)
        if plan_evaluation is not None:
            return round(float(plan_evaluation), 6)
        retrieval_confidence = getattr(getattr(execution_context.retrieval_response, "confidence", None), "confidence", None)
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
