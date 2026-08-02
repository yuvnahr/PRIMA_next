"""Integrated PRIMA-NEXT runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, cast

from affect.affect_engine import DynamicAffectEngine
from llm.llm_client import LLMClient
from llm.provider import ProviderError
from llm.response_parser import extract_answer
from memory.embedding_pipeline import current_embedding_metadata
from memory.maintenance.importance_types import MemoryAdmissionDecision
from memory.maintenance.memory_importance import MemoryImportanceEngine
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository, MemoryRepository
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_controller import RetrievalController
from memory.retrieval.retrieval_request import RetrievalRequest
from reasoning.config import reasoning_budget
from reasoning.config import reasoning_mode as configured_reasoning_mode
from reasoning.controller import ReasoningController
from reasoning.models import AnswerResult, ReasoningMode, ReasoningRequest
from reflection.reasoning_reflection_adapter import ReasoningReflectionAdapter
from reflection.reflection_engine import ReflectionEngine
from runtime.context_builder import AnswerContext, RuntimeContextBuilder
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
        self.retrieval_controller = RetrievalController(
            self.memory_repository,
            candidate_pool_size=int(os.getenv("PRIMA_RETRIEVAL_CANDIDATE_POOL_SIZE", "30")),
        )
        self.reasoning_controller = ReasoningController(
            reflection_advisor=ReasoningReflectionAdapter(self.reflection_engine)
        )
        self.workflow = workflow or PrimaWorkflow.from_controllers(
            affect_engine=self.affect_engine,
            retrieval_controller=self.retrieval_controller,
            reflection_engine=self.reflection_engine,
        )
        self.log_path = Path(log_path)
        self.logger = self._build_logger(self.log_path)

    def reset(self, mode: str = "isolated", preserve_repository: bool = True) -> None:
        """Reset runtime conversation state without deleting long-term memory by default."""

        if mode not in {"isolated", "persistent"}:
            raise ValueError("Runtime reset mode must be 'isolated' or 'persistent'.")
        if mode == "isolated" and not preserve_repository:
            self.memory_repository = InMemoryMemoryRepository()
            self.memory_importance_engine = MemoryImportanceEngine(self.memory_repository)
            self.retrieval_controller = RetrievalController(
                self.memory_repository,
                candidate_pool_size=int(os.getenv("PRIMA_RETRIEVAL_CANDIDATE_POOL_SIZE", "30")),
            )
            self.reasoning_controller = ReasoningController(
                reflection_advisor=ReasoningReflectionAdapter(self.reflection_engine)
            )
            self.workflow = PrimaWorkflow.from_controllers(
                affect_engine=self.affect_engine,
                retrieval_controller=self.retrieval_controller,
                reflection_engine=self.reflection_engine,
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
        """Answer through the canonical bounded evidence-acquisition controller."""

        started = time.perf_counter()
        top_k = int(top_k) if top_k else int(os.getenv("PRIMA_RETRIEVAL_TOP_K", "5"))
        max_context_tokens = int(max_context_tokens) if max_context_tokens else int(os.getenv("PRIMA_ANSWER_CONTEXT_TOKENS", "1600"))
        mode = ReasoningMode.DIAGNOSTIC if diagnostics else configured_reasoning_mode(reasoning_mode)
        request = ReasoningRequest(
            question=question,
            session_id=session_id or (context.session_id if context is not None else ""),
            mode=mode,
            budget=reasoning_budget(max_hops=max_hops, max_context_tokens=max_context_tokens),
        )
        result = self.reasoning_controller.answer(
            request,
            retrieve=lambda query: self.retrieval_controller.retrieve(RetrievalRequest(query=query, top_k=top_k)),
            synthesize=lambda prompt, evidence: self._synthesize_evidence(
                prompt, evidence, provider=provider, model=model, max_context_tokens=max_context_tokens
            ),
        )
        result.answer_diagnostics.update(
            {
                "question": question,
                "retrieved_memory_ids": [item.source_id for item in result.evidence_references],
                "retrieved_memories": [
                    {
                        "id": item.source_id,
                        "text": item.text,
                        "retrieval_score": item.retrieval_score,
                    }
                    for item in result.evidence_references
                ],
                "retrieval_scores": [item.retrieval_score for item in result.evidence_references],
                "retrieval_confidence": result.confidence,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "errors": list(result.errors),
                "embedding": current_embedding_metadata(),
            }
        )
        if context is not None:
            context.retrieved_memories = result.retrieved_memories
            context.confidence_score = result.confidence
        self._log_answer(question, {
            "retrieved_memory_ids": [item.source_id for item in result.evidence_references],
            "context_length": result.answer_diagnostics.get("context_tokens", 0),
            "llm_used": result.answer_diagnostics.get("llm_used", False),
            "latency_ms": 0.0,
            "reflection_used": False,
            "provider": result.answer_diagnostics.get("provider", ""),
            "model": result.answer_diagnostics.get("model", ""),
            "errors": list(result.errors),
        })
        return result

    def ingest_document(self, text: str, metadata: dict[str, Any] | None = None) -> MemoryNote:
        """Store one caller-supplied document through the public runtime boundary."""
        if not str(text).strip():
            raise ValueError("Document text must not be empty.")
        note = MemoryNote.create(content=str(text), memory_type=MemoryType.SEMANTIC, context={"source": "document_ingestion", **dict(metadata or {})})
        return self.memory_repository.add(note)

    def _synthesize_evidence(
        self,
        question: str,
        evidence: tuple[Any, ...],
        *,
        provider: str | None,
        model: str | None,
        max_context_tokens: int,
    ) -> tuple[str, bool, tuple[str, ...], dict[str, Any]]:
        """Reuse the established bounded context and LLM answer path after acquisition."""

        answer_context = RuntimeContextBuilder(max_context_tokens).build(question, evidence)
        errors: list[str] = []
        llm_used = False
        raw_model_response = ""
        selected_memory_ids: list[str] = []
        structured_answer_valid = False
        prompt: str | None = None
        llm_metadata: dict[str, Any] = {}
        inference_settings = self._inference_settings(
            provider=cast(str, provider or os.getenv("PRIMA_LLM_PROVIDER", "ollama")),
            model=cast(str, model or os.getenv("PRIMA_LLM_MODEL", "")),
            answer_context=answer_context,
        )
        if not answer_context.memories:
            answer = self._insufficient_information_answer(question, reason="no_retrieved_memories")
        else:
            prompt = self._build_answer_prompt(question, answer_context)
            self._log_inference_settings(inference_settings)
            try:
                llm_response = LLMClient(provider_name=str(inference_settings["provider"])).chat(
                    prompt=prompt,
                    model=str(inference_settings["model"]),
                    temperature=float(inference_settings["temperature"]),
                    max_tokens=int(inference_settings["num_predict"]),
                    system_prompt=self._answer_system_prompt(),
                    response_format=self._answer_response_schema(),
                )
                raw_model_response = llm_response.text.strip()
                llm_used = bool(raw_model_response)
                if isinstance(llm_response.raw, dict):
                    llm_metadata = {
                        key: llm_response.raw.get(key)
                        for key in (
                            "model",
                            "thinking",
                            "done",
                            "done_reason",
                            "total_duration",
                            "load_duration",
                            "prompt_eval_count",
                            "eval_count",
                        )
                    }
                if llm_metadata.get("done_reason") == "length":
                    errors.append("LLM answer was truncated at the generation limit.")
                answer, selected_memory_ids = self._parse_structured_answer(raw_model_response, answer_context)
                structured_answer_valid = True
            except (json.JSONDecodeError, TypeError, KeyError, ProviderError, RuntimeError, ValueError) as exc:
                errors.append(str(exc))
                answer = self._parse_partial_answer(raw_model_response) if raw_model_response.startswith("{") else raw_model_response
                if not answer:
                    answer = self._extractive_answer(answer_context)
        if answer.strip() == question.strip():
            errors.append("Answer matched question text; replaced with insufficient-information response.")
            answer = self._insufficient_information_answer(question, reason="answer_echo_guard")
        return answer, llm_used, tuple(errors), {
            "llm_used": llm_used,
            "context_tokens": answer_context.token_count,
            "memory_ids_used": [memory.memory_id for memory in answer_context.memories],
            "selected_memory_ids": selected_memory_ids,
            "structured_answer_valid": structured_answer_valid,
            "raw_model_response": raw_model_response,
            "raw_response": raw_model_response or None,
            "prompt": prompt,
            "llm_metadata": llm_metadata,
            "generation_settings": dict(inference_settings),
            "provider": provider or os.getenv("PRIMA_LLM_PROVIDER", "ollama"),
            "model": model or os.getenv("PRIMA_LLM_MODEL", ""),
        }

    def _parse_structured_answer(self, raw: str, answer_context: AnswerContext) -> tuple[str, list[str]]:
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("insufficient_information"), bool):
            raise ValueError("LLM returned an invalid structured answer.")
        answer = payload.get("answer")
        if payload["insufficient_information"]:
            if answer is not None:
                raise ValueError("Insufficient-information answer must be null.")
            return self._insufficient_information_answer(answer_context.question, reason="model_abstention"), []
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Structured answer must contain a non-empty answer string.")
        labels = payload.get("evidence")
        if not isinstance(labels, list) or not labels or any(not isinstance(label, str) for label in labels):
            raise ValueError("Structured answer evidence must be a list of context labels.")
        label_map = {f"M{index}": memory.memory_id for index, memory in enumerate(answer_context.memories, 1)}
        unknown = [label for label in labels if label not in label_map]
        if unknown:
            raise ValueError(f"Structured answer cited unknown context labels: {', '.join(unknown)}")
        return answer.strip(), list(dict.fromkeys(label_map[label] for label in labels))

    @staticmethod
    def _parse_partial_answer(raw: str) -> str:
        key = raw.find('"answer"')
        colon = raw.find(":", key + 8) if key >= 0 else -1
        if colon < 0:
            return ""
        try:
            answer, _ = json.JSONDecoder().raw_decode(raw[colon + 1:].lstrip())
        except json.JSONDecodeError:
            return ""
        return answer.strip() if isinstance(answer, str) else ""

    def _answer_system_prompt(self) -> str:
        return (
            "You are a factual question-answering engine. Return only JSON matching the supplied schema. "
            "The answer must be the shortest exact answer span, normally one to eight words, never a sentence or explanation. "
            "Use the canonical singular form for a category, profession, nationality, or type. "
            "Cite the smallest sufficient set of [M#] labels; for comparisons or shared properties, cite evidence for each subject."
        )

    def _answer_response_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {"type": ["string", "null"], "maxLength": 80},
                "evidence": {"type": "array", "items": {"type": "string", "pattern": "^M[1-9][0-9]*$"}, "minItems": 1, "maxItems": 4},
                "insufficient_information": {"type": "boolean"},
            },
            "required": ["answer", "evidence", "insufficient_information"],
            "additionalProperties": False,
        }

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
            created_notes, admission_decision = self._persist_turn_memory(user_input, execution_context, runtime_context)
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
            affective_state=(affect_update.to_dict() if (affect_update is not None and hasattr(affect_update, "to_dict")) else {}),
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
            "memory_creation_count": metrics.memory_creation_count,
            "memory_admission": dict(result.memory_admission),
            "reflection_events": metrics.reflection_trigger_count,
            "confidence_score": metrics.confidence_score,
            "latency_ms": metrics.latency_ms,
            "errors": list(result.errors),
        }
        self.logger.info(json.dumps(payload, sort_keys=True))

    def _build_answer_prompt(self, question: str, answer_context: AnswerContext) -> str:
        return (
            "Use only the retrieved context below.\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{answer_context.context_text}\n\n"
            "Return JSON matching the supplied schema. Put only the shortest exact answer in `answer`; "
            "use null only when the context is insufficient. Cite the smallest sufficient set of context labels in `evidence`."
        )

    def _extractive_answer(self, answer_context: AnswerContext) -> str:
        if not answer_context.memories:
            return self._insufficient_information_answer(answer_context.question, reason="no_context")
        best = max(answer_context.memories, key=lambda memory: memory.retrieval_score)
        return (
            "Based on retrieved memory: "
            f"{best.text}"
        )

    def _insufficient_information_answer(self, question: str, reason: str) -> str:
        return json.dumps(
            {
                "answer": None,
                "insufficient_information": True,
                "reason": reason,
                "question": question,
            },
            sort_keys=True,
        )

    def _answer_diagnostics(
        self,
        question: str,
        answer_context: AnswerContext,
        retrieval_response: Any,
        llm_used: bool,
        latency_ms: float,
        provider: str,
        model: str,
        errors: tuple[str, ...],
        inference_settings: dict[str, Any],
        prompt: str | None,
        raw_response: str | None,
        llm_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        retrieval_diagnostics = dict(getattr(retrieval_response, "diagnostics", {}))
        diagnostics = {
            "question": question,
            "retrieved_memory_ids": [memory.memory_id for memory in answer_context.memories],
            "retrieved_memories": [
                {
                    "id": memory.memory_id,
                    "text": memory.text,
                    "timestamp": memory.timestamp,
                    "importance_score": memory.importance_score,
                    "retrieval_score": memory.retrieval_score,
                }
                for memory in answer_context.memories
            ],
            "retrieval_scores": [memory.retrieval_score for memory in answer_context.memories],
            "context_length": answer_context.token_count,
            "llm_used": llm_used,
            "latency_ms": latency_ms,
            "reflection_used": answer_context.reflection_used,
            "memory_ids_used": [memory.memory_id for memory in answer_context.memories],
            "graph_links_traversed": list(answer_context.graph_links_traversed),
            "importance_scores": [memory.importance_score for memory in answer_context.memories],
            "retrieval_confidence": retrieval_response.confidence.confidence,
            "retrieval_confidence_components": retrieval_response.confidence.to_dict(),
            "expanded_query": dict(retrieval_diagnostics.get("expanded_query", {})),
            "entities": list(retrieval_diagnostics.get("entities", [])),
            "relations": list(retrieval_diagnostics.get("relations", [])),
            "temporal_constraints": list(retrieval_diagnostics.get("temporal_constraints", [])),
            "retrieval_stages": {
                stage: _compact_candidates(retrieval_diagnostics.get(stage, ()))
                for stage in ("dense_top30", "sparse_top30", "fused_top30", "reranked_top30", "final_candidates")
            },
            "stored_source_turn_ids": sorted(
                {
                    str(note.context["source_turn_id"])
                    for note in self.memory_repository.list()
                    if note.context.get("source_turn_id")
                }
            ),
            "embedding": current_embedding_metadata(),
            "context_tokens": answer_context.token_count,
            "provider": provider,
            "model": model,
            "errors": list(errors),
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_answer": extract_answer(raw_response) if raw_response is not None else None,
            "llm_metadata": llm_metadata,
            "generation_settings": dict(inference_settings),
        }
        diagnostics["failure_type"] = self._classify_answer_failure(diagnostics, errors, llm_used)
        return diagnostics


    def _classify_answer_failure(self, diagnostics: dict[str, Any], errors: tuple[str, ...], llm_used: bool) -> str | None:
        if errors:
            return "generation_error"
        retrieved = diagnostics.get("retrieved_memory_ids", [])
        if not retrieved:
            return "retrieval_miss"
        if not llm_used:
            return "generation_error"
        return None

    def _log_answer(self, question: str, diagnostics: dict[str, Any]) -> None:
        payload = {
            "answer_event": {
                "retrieved_memory_count": len(diagnostics.get("retrieved_memory_ids", ())),
                "context_length": diagnostics.get("context_length", 0),
                "llm_used": diagnostics.get("llm_used", False),
                "latency_ms": diagnostics.get("latency_ms", 0.0),
                "reflection_used": diagnostics.get("reflection_used", False),
                "provider": diagnostics.get("provider", ""),
                "model": diagnostics.get("model", ""),
                "errors": diagnostics.get("errors", []),
            }
        }
        self.logger.info(json.dumps(payload, sort_keys=True))

    def _inference_settings(
        self,
        provider: str,
        model: str,
        answer_context: AnswerContext,
    ) -> dict[str, Any]:
        return {
            "provider": provider,
            "model": model,
            "temperature": self._env_float("PRIMA_ANSWER_TEMPERATURE", 0.0),
            "top_p": self._env_float("PRIMA_ANSWER_TOP_P", 0.8),
            "top_k": self._env_int("PRIMA_ANSWER_TOP_K", 40),
            "repeat_penalty": self._env_float("PRIMA_ANSWER_REPEAT_PENALTY", 1.1),
            "seed": self._env_int("PRIMA_ANSWER_SEED", 13),
            "num_predict": self._env_int("PRIMA_ANSWER_MAX_TOKENS", 128),
            "context_tokens": answer_context.token_count,
            "retrieved_memories": len(answer_context.memories),
        }

    def _log_inference_settings(self, settings: dict[str, Any]) -> None:
        if not self._debug_inference_enabled():
            return
        self.logger.info(json.dumps({"inference_settings": settings}, sort_keys=True))

    def _debug_inference_enabled(self) -> bool:
        return os.getenv("PRIMA_DEBUG_INFERENCE", "").lower() in {"1", "true", "yes"}

    def _env_float(self, name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            return default

    def _env_int(self, name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            return default

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


def _compact_candidates(candidates: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": item.get("id"),
            "source_session_id": item.get("source_session_id"),
            "source_turn_id": item.get("source_turn_id"),
            "score": item.get("score"),
            "strategy_scores": dict(item.get("strategy_scores", {})),
        }
        for item in candidates
    ]


