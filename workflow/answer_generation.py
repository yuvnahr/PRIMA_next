"""Workflow-owned answer generation using an injected model client."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from llm.llm_client import LLMClient
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
        }


@dataclass(slots=True)
class AnswerGenerationController:
    """Generate one answer from workflow-routed evidence, plan and state."""

    llm_client: LLMClient
    phase: WorkflowPhase = WorkflowPhase.ANSWER_GENERATION

    async def execute(self, context: ExecutionContext) -> GenerationResult:
        """Call the injected client or return an explicit evidence abstention."""

        task_kind = str(context.metadata.get("task_kind", "conversation"))
        reasoning = context.reasoning_result if isinstance(context.reasoning_result, AnswerResult) else None
        provider = str(getattr(self.llm_client, "provider_name", ""))
        model = str(context.metadata.get("model", ""))
        if task_kind == "factual_qa" and reasoning is not None:
            if reasoning.status is SufficiencyStatus.ERROR:
                return GenerationResult(
                    GenerationOutcome.FAILED,
                    errors=reasoning.errors or ("Evidence acquisition failed.",),
                    provider=provider,
                    model=model,
                )
            if reasoning.status is not SufficiencyStatus.SUFFICIENT:
                return GenerationResult(
                    GenerationOutcome.ABSTAINED,
                    text="I do not have sufficient evidence to answer this question.",
                    provider=provider,
                    model=model,
                )

        prompt, label_map = self._prompt(context, reasoning)
        factual = task_kind == "factual_qa"
        try:
            response = await asyncio.to_thread(
                self.llm_client.chat,
                prompt=prompt,
                model=model or None,
                temperature=float(context.metadata.get("temperature", 0.0)),
                max_tokens=int(context.metadata.get("max_tokens", 128)),
                system_prompt=self._system_prompt(factual),
                response_format=self._response_schema() if factual else None,
            )
            raw = response.text.strip()
            return self._parse(raw, context.user_input, label_map, factual, provider, model)
        except (ProviderError, RuntimeError, TypeError, ValueError) as exc:
            return GenerationResult(
                GenerationOutcome.FAILED,
                errors=(str(exc),),
                provider=provider,
                model=model,
                llm_called=True,
            )

    def _prompt(
        self,
        context: ExecutionContext,
        reasoning: AnswerResult | None,
    ) -> tuple[str, dict[str, str]]:
        evidence = reasoning.evidence_references if reasoning is not None else ()
        label_map = {f"E{index}": item.source_id for index, item in enumerate(evidence, 1)}
        evidence_text = "\n".join(
            f"[{label}] {item.text}" for label, item in zip(label_map, evidence, strict=True)
        ) or "(none selected)"
        plan = context.plan.to_dict() if context.plan is not None and hasattr(context.plan, "to_dict") else None
        state = {
            "goal": context.cognitive_state.goal_state,
            "task": context.cognitive_state.task_state,
            "confidence": context.cognitive_state.confidence_state,
        }
        return (
            f"Request: {context.user_input}\n"
            f"Selected evidence:\n{evidence_text}\n"
            f"Plan: {json.dumps(plan, default=str, sort_keys=True)}\n"
            f"State: {json.dumps(state, default=str, sort_keys=True)}",
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
        )

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
