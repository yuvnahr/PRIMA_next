"""Workflow-owned answer generation using an injected model client."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from llm.generation_config import FallbackPolicy, GenerationConfig, StructuredOutputMode
from llm.llm_client import LLMClient
from llm.prompt_builder import PromptBuilder, PromptEvidence
from llm.provider import ProviderError
from reasoning.models import AnswerResult, SufficiencyStatus
from workflow.execution_context import ExecutionContext
from workflow.workflow_state import WorkflowPhase


class GenerationOutcome(Enum):
    """Typed terminal outcome of model generation."""

    ANSWERED = "answered"
    ABSTAINED = "abstained"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Typed answer-generation output consumed by result shaping."""

    outcome: GenerationOutcome
    text: str | None = None
    selected_source_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    provider: str = ""
    model: str = ""
    llm_called: bool = False
    latency_ms: float = 0.0
    usage: dict[str, int] | None = None
    fallback_policy: str = FallbackPolicy.ABSTAIN.value
    fallback_used: bool = False
    fallback_reason: str | None = None
    provider_capabilities: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible generation record."""

        return {
            "outcome": self.outcome.value,
            "text": self.text,
            "selected_source_ids": list(self.selected_source_ids),
            "errors": list(self.errors),
            "provider": self.provider,
            "model": self.model,
            "llm_called": self.llm_called,
            "latency_ms": self.latency_ms,
            "usage": dict(self.usage or {}),
            "fallback_policy": self.fallback_policy,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "provider_capabilities": dict(self.provider_capabilities or {}),
        }


@dataclass(slots=True)
class AnswerGenerationController:
    """Generate one answer from workflow-routed evidence, plan and state."""

    llm_client: LLMClient
    default_generation_config: GenerationConfig | None = None
    phase: WorkflowPhase = WorkflowPhase.ANSWER_GENERATION

    async def execute(self, context: ExecutionContext) -> GenerationResult:
        """Call the injected client or return an explicit evidence abstention."""

        task_kind = str(context.metadata.get("task_kind", "conversation"))
        reasoning = context.reasoning_result if isinstance(context.reasoning_result, AnswerResult) else None
        provider = str(getattr(self.llm_client, "provider_name", ""))
        config = context.metadata.get("generation_config")
        if not isinstance(config, GenerationConfig):
            config = self.default_generation_config
        if config is None:
            return GenerationResult(
                GenerationOutcome.FAILED,
                errors=("A typed GenerationConfig is required for answer generation.",),
                provider=provider,
            )
        factual = task_kind == "factual_qa"
        if factual and config.structured_output is not StructuredOutputMode.JSON_SCHEMA:
            config = config.with_overrides(structured_output=StructuredOutputMode.JSON_SCHEMA)
        model = config.model
        if task_kind == "factual_qa" and reasoning is not None:
            if reasoning.status is SufficiencyStatus.ERROR:
                return GenerationResult(
                    GenerationOutcome.FAILED,
                    errors=reasoning.errors or ("Evidence acquisition failed.",),
                    provider=provider,
                    model=model,
                    fallback_policy=config.fallback_policy.value,
                    provider_capabilities=dict(getattr(self.llm_client, "capabilities", {})),
                )
            if reasoning.status is not SufficiencyStatus.SUFFICIENT:
                return GenerationResult(
                    GenerationOutcome.ABSTAINED,
                    text="I do not have sufficient evidence to answer this question.",
                    provider=provider,
                    model=model,
                    fallback_policy=config.fallback_policy.value,
                    provider_capabilities=dict(getattr(self.llm_client, "capabilities", {})),
                )

        prompt, label_map = self._prompt(context, reasoning, factual)
        started = time.perf_counter()
        context.metadata["model_call_count"] = int(context.metadata.get("model_call_count", 0)) + 1
        try:
            response = await asyncio.to_thread(
                self.llm_client.chat,
                prompt=prompt.user_prompt,
                generation_config=config,
                system_prompt=prompt.system_policy,
                response_format=prompt.response_schema,
                response_schema=prompt.response_schema,
            )
            raw = response.text.strip()
            aggregate_usage = dict(context.metadata.get("model_usage", {}))
            for key, value in (response.usage or {}).items():
                aggregate_usage[str(key)] = int(aggregate_usage.get(str(key), 0)) + int(value)
            context.metadata["model_usage"] = aggregate_usage
            return self._parse(
                raw,
                context.user_input,
                label_map,
                factual,
                provider,
                model,
                config,
                round((time.perf_counter() - started) * 1000, 3),
                response.usage,
            )
        except (ProviderError, RuntimeError, TypeError, ValueError) as exc:
            return self._fallback(
                reasoning,
                config,
                str(exc),
                round((time.perf_counter() - started) * 1000, 3),
            )

    def _prompt(
        self,
        context: ExecutionContext,
        reasoning: AnswerResult | None,
        factual: bool,
    ) -> tuple[Any, dict[str, str]]:
        evidence = reasoning.evidence_references if reasoning is not None else ()
        label_map = {f"E{index}": item.source_id for index, item in enumerate(evidence, 1)}
        plan = context.plan.to_dict() if context.plan is not None and hasattr(context.plan, "to_dict") else None
        state = {
            "goal": context.cognitive_state.goal_state,
            "task": context.cognitive_state.task_state,
            "confidence": context.cognitive_state.confidence_state,
        }
        schema = self._response_schema() if factual else None
        return (
            PromptBuilder.build(
                system_policy=self._system_prompt(factual),
                user_input=context.user_input,
                evidence=(
                    PromptEvidence(label, item.source_id, item.text)
                    for label, item in zip(label_map, evidence, strict=True)
                ),
                tool_results=tuple(str(item) for item in context.metadata.get("tool_results", ())),
                runtime_context={"plan": plan, "state": state},
                response_schema=schema,
            ),
            label_map,
        )

    def _parse(
        self,
        raw: str,
        request_text: str,
        label_map: dict[str, str],
        factual: bool,
        provider: str,
        model: str,
        config: GenerationConfig,
        latency_ms: float,
        usage: dict[str, int] | None,
    ) -> GenerationResult:
        if not raw:
            raise ValueError("LLM returned an empty response.")
        selected: tuple[str, ...] = ()
        answer = raw
        if factual:
            payload = json.loads(raw)
            if not isinstance(payload, dict) or not isinstance(payload.get("insufficient_information"), bool):
                raise ValueError("LLM returned an invalid factual answer.")
            if payload["insufficient_information"]:
                return GenerationResult(
                    GenerationOutcome.ABSTAINED,
                    text="I do not have sufficient evidence to answer this question.",
                    provider=provider,
                    model=model,
                    llm_called=True,
                    latency_ms=latency_ms,
                    usage=usage,
                    fallback_policy=config.fallback_policy.value,
                    provider_capabilities=dict(getattr(self.llm_client, "capabilities", {})),
                )
            answer_value = payload.get("answer")
            if not isinstance(answer_value, str) or not answer_value.strip():
                raise ValueError("LLM factual answer is empty.")
            labels = payload.get("evidence", [])
            if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
                raise ValueError("LLM evidence labels are invalid.")
            unknown = [label for label in labels if label not in label_map]
            if unknown:
                raise ValueError(f"LLM cited unknown evidence labels: {', '.join(unknown)}")
            answer = answer_value.strip()
            selected = tuple(dict.fromkeys(label_map[label] for label in labels))
        if answer.strip() == request_text.strip():
            raise ValueError("LLM answer echoed the request.")
        return GenerationResult(
            GenerationOutcome.ANSWERED,
            text=answer.strip(),
            selected_source_ids=selected,
            provider=provider,
            model=model,
            llm_called=True,
            latency_ms=latency_ms,
            usage=usage,
            fallback_policy=config.fallback_policy.value,
            provider_capabilities=dict(getattr(self.llm_client, "capabilities", {})),
        )

    def _fallback(
        self,
        reasoning: AnswerResult | None,
        config: GenerationConfig,
        reason: str,
        latency_ms: float,
    ) -> GenerationResult:
        def result(
            outcome: GenerationOutcome,
            *,
            text: str | None = None,
            selected_source_ids: tuple[str, ...] = (),
            fallback_used: bool = False,
        ) -> GenerationResult:
            return GenerationResult(
                outcome=outcome,
                text=text,
                selected_source_ids=selected_source_ids,
                errors=(reason,),
                provider=config.provider,
                model=config.model,
                llm_called=True,
                latency_ms=latency_ms,
                fallback_policy=config.fallback_policy.value,
                fallback_used=fallback_used,
                fallback_reason=reason,
                provider_capabilities=dict(getattr(self.llm_client, "capabilities", {})),
            )
        if config.fallback_policy is FallbackPolicy.EXTRACTIVE and reasoning is not None:
            evidence = reasoning.evidence_references
            if evidence:
                return result(
                    GenerationOutcome.ANSWERED,
                    text=evidence[0].text.strip(),
                    selected_source_ids=(evidence[0].source_id,),
                    fallback_used=True,
                )
        if config.fallback_policy is FallbackPolicy.ABSTAIN:
            return result(
                GenerationOutcome.ABSTAINED,
                text="I do not have sufficient evidence to answer this question.",
                fallback_used=True,
            )
        return result(GenerationOutcome.FAILED)

    @staticmethod
    def _system_prompt(factual: bool) -> str:
        if factual:
            return "Answer concisely. Return only JSON matching the supplied schema and cite selected [E#] labels."
        return "Respond directly to the user. Never repeat the request as the answer."

    @staticmethod
    def _response_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {"type": ["string", "null"]},
                "evidence": {"type": "array", "items": {"type": "string", "pattern": "^E[1-9][0-9]*$"}},
                "insufficient_information": {"type": "boolean"},
            },
            "required": ["answer", "evidence", "insufficient_information"],
            "additionalProperties": False,
        }
