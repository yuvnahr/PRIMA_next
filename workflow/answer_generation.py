"""Workflow-owned answer generation using an injected model client."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from llm.generation_config import FallbackPolicy, GenerationConfig, StructuredOutputMode
from llm.llm_client import LLMClient
from llm.prompt_builder import PromptBuilder, PromptEvidence
from llm.provider import ProviderError
from llm.request_budget import RequestBudget, RequestBudgetExceeded, prepaid_call
from reasoning.evidence_audit import source_turn_id
from reasoning.models import AnswerResult, SufficiencyStatus
from workflow.execution_context import ExecutionContext
from workflow.workflow_state import WorkflowPhase


def _prompt_context(plan: dict[str, Any] | None, state: dict[str, Any], labels: dict[str, str]) -> dict[str, Any]:
    """Present stable identity references without changing storage or human text."""
    aliases = {source: label for label, source in labels.items()}
    if plan:
        revision = plan.get("revision", 0)
        aliases[plan["plan_id"]] = f"P{revision}"
        if plan.get("previous_plan_id"):
            aliases[plan["previous_plan_id"]] = f"P{revision - 1}"
        for index, action in enumerate(plan.get("actions", ()), 1):
            aliases[action["action_id"]] = f"A{index}"
            if action["action_type"] == "integrate_memory":
                for source in action["inputs"]:
                    aliases.setdefault(source, f"M{len(aliases) + 1}")

    def references(value: Any, identity: bool = False) -> Any:
        if isinstance(value, dict):
            return {key: references(item, key.endswith(("_id", "_ids")) or key == "inputs")
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [references(item, identity) for item in value]
        return aliases.get(value, value) if identity and isinstance(value, str) else value

    return {"plan": references(plan), "state": references(state)}


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
    failure_kind: str | None = None
    blocked_reason: str | None = None
    validation_errors: tuple[str, ...] = ()
    rejected_output: str | None = None
    raw_output: str | None = None

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
            "failure_kind": self.failure_kind,
            "blocked_reason": self.blocked_reason,
            "validation_errors": list(self.validation_errors),
            "rejected_output": self.rejected_output,
            "raw_output": self.raw_output,
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
            return GenerationResult(
                GenerationOutcome.FAILED,
                errors=("Factual QA requires a precomputed JSON-schema GenerationConfig.",),
                provider=provider,
                model=config.model,
                fallback_policy=config.fallback_policy.value,
            )
        model = config.model
        if context.request_budget is None:
            limits = reasoning.effective_budget if reasoning is not None else None
            context.request_budget = RequestBudget(
                limits.max_llm_calls if limits else int(context.metadata.get("max_llm_calls", 4)),
                limits.time_budget_seconds if limits else float(context.metadata.get("time_budget_seconds", 15.0)),
                limits.answer_reserve_seconds if limits else float(context.metadata.get("answer_reserve_seconds", 3.0)))
        budget = context.request_budget

        def blocked(reason: str, *, called: bool = False) -> GenerationResult:
            return GenerationResult(GenerationOutcome.ABSTAINED,
                text="I do not have sufficient evidence to answer this question.", provider=provider, model=model,
                fallback_policy=config.fallback_policy.value, blocked_reason=reason, llm_called=called)

        if context.hard_error:
            return blocked("hard_error")
        denial = budget.denial_reason()
        if denial:
            return blocked(denial)
        if factual and reasoning is None and str(context.metadata.get("profile", "")) != "model_only":
            return blocked("no_eligible_evidence")
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
            if reasoning.errors:
                return blocked("hard_error")
            if reasoning.status in {SufficiencyStatus.CONTRADICTORY, SufficiencyStatus.AMBIGUOUS, SufficiencyStatus.UNANSWERABLE}:
                return blocked(reasoning.status.value)
            if not (reasoning.answer_attempt_permitted is True or (
                reasoning.answer_attempt_permitted is None and reasoning.status is SufficiencyStatus.SUFFICIENT
            )):
                return blocked("answer_not_permitted")
            eligible = [item for item in reasoning.evidence_references if item.text.strip() and item.provenance.get("evidence_eligible", True)]
            if not eligible:
                return blocked("no_eligible_evidence")
            if sum(len(item.text.split()) for item in eligible) > reasoning.effective_budget.max_context_tokens:
                return blocked("context_budget")

        bounded = factual and reasoning is not None and reasoning.status is not SufficiencyStatus.SUFFICIENT
        if bounded and budget.bounded_answer_attempts:
            return context.generation_result if isinstance(context.generation_result, GenerationResult) else blocked("answer_attempt_exhausted")

        prompt, label_map = self._prompt(context, reasoning, factual)
        started = time.perf_counter()
        failure_kind = "provider"
        raw: str | None = None
        try:
            budget.reserve_call()
            if bounded:
                budget.bounded_answer_attempts += 1
            call_config = config.with_overrides(timeout_seconds=min(config.timeout_seconds, budget.remaining_seconds),
                                                retries=0 if bounded else config.retries)
            def chat() -> Any:
                if budget.remaining_seconds <= 0:
                    raise RequestBudgetExceeded("time_budget")
                return self.llm_client.chat(prompt=prompt.user_prompt, generation_config=call_config,
                    system_prompt=prompt.system_policy, response_format=prompt.response_schema,
                    response_schema=prompt.response_schema)

            with prepaid_call(budget):
                response = await asyncio.wait_for(asyncio.to_thread(chat), timeout=budget.remaining_seconds)
            if budget.remaining_seconds <= 0:
                return blocked("time_budget", called=True)
            raw = response.text.strip()
            aggregate_usage = dict(context.metadata.get("model_usage", {}))
            for key, value in (response.usage or {}).items():
                aggregate_usage[str(key)] = int(aggregate_usage.get(str(key), 0)) + int(value)
            context.metadata["model_usage"] = aggregate_usage
            failure_kind = "parse"
            parsed = self._parse(
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
            return replace(parsed, raw_output=raw)
        except TimeoutError:
            return blocked("time_budget", called=True)
        except RequestBudgetExceeded as exc:
            return blocked(str(exc), called=budget.llm_calls > 0)
        except (ProviderError, RuntimeError, TypeError, ValueError) as exc:
            fallback = self._fallback(
                reasoning,
                config,
                str(exc),
                round((time.perf_counter() - started) * 1000, 3),
                failure_kind=failure_kind,
            )
            return replace(fallback, raw_output=raw)
        finally:
            context.metadata["model_call_count"] = budget.llm_calls

    def _prompt(
        self,
        context: ExecutionContext,
        reasoning: AnswerResult | None,
        factual: bool,
    ) -> tuple[Any, dict[str, str]]:
        evidence = tuple(item for item in reasoning.evidence_references if item.text.strip()
                         and item.provenance.get("evidence_eligible", True)) if reasoning is not None else ()
        label_map = {f"E{index}": item.source_id for index, item in enumerate(evidence, 1)}
        conversation_id = str(context.metadata.get("session_id", ""))
        plan = context.plan.to_dict() if context.plan is not None and hasattr(context.plan, "to_dict") else None
        state = {
            "goal": dict(context.cognitive_state.goal_state),
            "task": dict(context.cognitive_state.task_state),
            "confidence": dict(context.cognitive_state.confidence_state),
        }
        schema = self._response_schema() if factual else None
        prompt = PromptBuilder.build(
            system_policy=self._system_prompt(factual),
            user_input=context.user_input,
            evidence=(
                PromptEvidence(label, label, item.text)
                for label, item in zip(label_map, evidence, strict=True)
            ),
            tool_results=tuple(str(item) for item in context.metadata.get("tool_results", ())),
            runtime_context=_prompt_context(plan, state, label_map),
            response_schema=schema,
        )
        context.prompt_evidence = [
            {"memory_id": item.source_id, "source_turn_id": source_turn_id(item.provenance, conversation_id)}
            for item in evidence
        ]
        return prompt, label_map

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
            unknown.extend(label for label in re.findall(r"\[(E\d+)\]", answer_value) if label not in label_map or label not in labels)
            if unknown:
                raise ValueError(f"LLM cited unknown evidence labels: {', '.join(unknown)}")
            if label_map and not labels:
                return GenerationResult(GenerationOutcome.ABSTAINED,
                    text="I do not have sufficient evidence to answer this question.", provider=provider, model=model,
                    llm_called=True, latency_ms=latency_ms, usage=usage, fallback_policy=config.fallback_policy.value,
                    blocked_reason="missing_citations", validation_errors=("Factual answer must cite supplied evidence labels",),
                    rejected_output=raw)
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
        *, failure_kind: str = "generation",
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
                failure_kind=failure_kind,
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
            return ("Answer concisely using only the supplied evidence. Return only JSON matching the schema and cite selected [E#] labels. "
                    "If support is missing or contradictory, return insufficient_information=true. Never invent facts or citation labels. "
                    "Use explicit source date anchors to resolve omitted months/years only when unambiguous; do not equate every event date with the source date. "
                    "Include every supported requested list item, and compose multi-hop answers only from explicitly supported links.")
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
