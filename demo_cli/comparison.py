"""Paired input and independent histories; PRIMA still owns its own execution."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from config.runtime_config import RuntimeConfig
from config.settings import Settings, get_settings
from demo_cli.config import DemoConfig
from demo_cli.errors import safe_reason
from demo_cli.live_eval import (
    GROQ_JUDGE_MODEL,
    JUDGE_POLICY,
    JUDGE_SCHEMA,
    EvalUnavailable,
    PairedEvaluation,
    judge_prompt,
    parse_evaluation,
)
from demo_cli.session import DemoSession
from llm.generation_config import GenerationConfig, StructuredOutputMode
from llm.llm_client import LLMClient
from llm.prompt_builder import CONVERSATION_POLICY
from llm.provider import ProviderError
from monitoring.tracing import Tracing
from runtime import ExecutionProfile, ExecutionStatus, PrimaRequest, TaskKind

HISTORY_CHARACTER_LIMIT = 100_000


@dataclass(frozen=True)
class BranchResult:
    text: str
    status: str
    latency_ms: float
    usage: dict[str, int]
    calls: int
    trace_id: str | None = None


@dataclass(frozen=True)
class ComparisonTurn:
    comparison_id: str
    index: int
    base: BranchResult
    prima: BranchResult


class ComparisonSession:
    def __init__(self, config: DemoConfig, *, settings: Settings | None = None, llm_client: LLMClient | None = None,
                 observability: Tracing | None = None, runtime_config: RuntimeConfig | None = None) -> None:
        if config.persist or config.profile is not ExecutionProfile.PRIMA_FULL:
            raise ValueError("Comparison requires process-local storage and the prima profile.")
        client = llm_client or LLMClient(provider_name=config.generation.provider, settings=settings or get_settings())
        if not client.provider.supports_history:
            raise ValueError("Comparison requires an OpenAI-compatible provider with role-based history (e.g. Groq).")
        self.config = config
        self.comparison_id = f"compare-{uuid4().hex[:12]}"
        self.base_session_id = f"{self.comparison_id}-base"
        self.prima = DemoSession(replace(config, session_id=f"{config.session_id}-prima"), settings=settings,
                                 llm_client=client, observability=observability, runtime_config=runtime_config)
        self.base_history: list[dict[str, str]] = []
        self.turn_index = 0
        self.needs_reset = False
        self.latest_turn: ComparisonTurn | None = None
        self._eval_transcript: list[tuple[str, str, str]] = []
        self._evaluation: PairedEvaluation | None = None
        self._base_first = int(self.comparison_id[-1], 16) % 2 == 0
        self._judge = GenerationConfig(
            model=config.judge_model or (GROQ_JUDGE_MODEL if config.generation.provider == "groq" else config.generation.model),
            provider=config.generation.provider,
            max_output_tokens=2048,
            timeout_seconds=15,
            structured_output=StructuredOutputMode.JSON_SCHEMA,
        )

    def status_rows(self) -> tuple[tuple[str, str], ...]:
        return (*self.prima.status_rows(), ("Comparison", self.comparison_id), ("Base session", self.base_session_id),
                ("Base context", f"{len(self.base_history) // 2} complete turns; full history, no dropped messages"),
                ("History limit", f"{HISTORY_CHARACTER_LIMIT} characters including current prompt/policy"),
                ("Live judge", self._judge.model),
                ("Comparison state", "reset required after failure" if self.needs_reset else "ready"))

    def _metadata(self, branch: str) -> dict[str, Any]:
        return {"comparison_id": self.comparison_id, "comparison_turn": self.turn_index, "comparison_branch": branch}

    async def _base(self, text: str) -> BranchResult:
        client = self.prima.runtime.llm_client
        before = client.call_count
        started = time.perf_counter()
        request = PrimaRequest(task_kind=TaskKind.CONVERSATION, profile=ExecutionProfile.MODEL_ONLY,
                               session_id=self.base_session_id, input_text=text, metadata=self._metadata("base"))
        usage = {}
        with self.prima.observability.turn(request, name="base.turn", capture_content=self.config.trace_content) as observation:
            observation.set_attributes({"provider": self.config.generation.provider, "model": self.config.generation.model,
                                        "execution_mode": "direct_model_chat", "history_turns": len(self.base_history) // 2})
            try:
                response = await asyncio.to_thread(client.chat, text, generation_config=self.config.generation,
                                                   system_prompt=CONVERSATION_POLICY, history=tuple(self.base_history))
                usage = {key: response.usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                         if isinstance(response.usage, dict) and type(response.usage.get(key)) is int and response.usage[key] >= 0}
                if not response.text or not response.text.strip():
                    raise ValueError("The base model returned an empty answer.")
                answer, status = response.text, "completed"
                self.base_history.extend(({"role": "user", "content": text}, {"role": "assistant", "content": answer}))
                observation.content(output={"answer": answer, "status": status})
            except Exception as exc:
                answer, status = safe_reason(exc, provider=self.config.generation.provider), "failed"
                observation.fail(type(exc).__name__)
                observation.content(output={"status": status, "error_type": type(exc).__name__})
            observation.set_attributes({"status": status})
            trace_id = observation.trace_id
        return BranchResult(answer, status, round((time.perf_counter() - started) * 1000, 3),
                            usage, client.call_count - before, trace_id)

    async def _prima(self, text: str) -> BranchResult:
        client = self.prima.runtime.llm_client
        before = client.call_count
        started = time.perf_counter()
        self.prima.runtime.last_trace_id = None
        try:
            response = await self.prima.send(text, trace_metadata=self._metadata("prima"))
            status = response.status.value
            if response.status is ExecutionStatus.COMPLETED:
                answer = response.output_text or "No answer returned."
                if response.outcome and response.outcome.value != "answered":
                    status = response.outcome.value
            else:
                answer = safe_reason("; ".join(response.errors) or status, provider=self.config.generation.provider)
            usage = dict(response.diagnostics.model_usage)
        except Exception as exc:
            answer, status, usage = safe_reason(exc, provider=self.config.generation.provider), "failed", {}
        return BranchResult(answer, status, round((time.perf_counter() - started) * 1000, 3),
                            usage, client.call_count - before, self.prima.runtime.last_trace_id)

    async def send(self, text: str) -> ComparisonTurn:
        if self.needs_reset:
            raise ValueError("The branches have uneven histories after a failure. Use /reset before continuing.")
        if not text.strip():
            raise ValueError("Enter a nonempty comparison prompt.")
        size = len(CONVERSATION_POLICY) + len(text) + sum(len(item["content"]) for item in self.base_history)
        if size > HISTORY_CHARACTER_LIMIT:
            raise ValueError("Baseline history limit reached; no messages were dropped and neither branch ran. Use /reset.")
        self.turn_index += 1
        # Sequential calls share the same rate limiter and preserve readable foreground events.
        base = await self._base(text)
        prima = await self._prima(text)
        self.needs_reset = base.status == "failed" or prima.status in {"failed", "rejected", "cancelled", "not_implemented"}
        self.latest_turn = ComparisonTurn(self.comparison_id, self.turn_index, base, prima)
        self._eval_transcript.append((text, base.text, prima.text))
        self._evaluation = None
        return self.latest_turn

    async def evaluate_latest(self, announce: Callable[[str], None]) -> PairedEvaluation:
        turn = self.latest_turn
        if turn is None:
            raise EvalUnavailable("Send a paired turn before using /eval.")
        if any(branch.status not in {"completed", "abstained"} or not branch.text.strip()
               or branch.text == "No answer returned." for branch in (turn.base, turn.prima)):
            raise EvalUnavailable("Both branches must return an answer to score this pair.")
        if self._evaluation is not None:
            return self._evaluation

        client = self.prima.runtime.llm_client
        if StructuredOutputMode.JSON_SCHEMA not in client.provider.capabilities.structured_output:
            raise EvalUnavailable("The current provider adapter does not support the judge's JSON schema.")
        base_first = self._base_first
        secrets = self.prima.observability.secrets
        prompt = judge_prompt(self._eval_transcript, base_first=base_first, secrets=secrets)
        announce(f"Evaluating paired turn {turn.index} with {self._judge.model}...")
        try:
            response = await asyncio.to_thread(client.chat, prompt, generation_config=self._judge,
                                               system_prompt=JUDGE_POLICY, response_schema=JUDGE_SCHEMA)
        except (ProviderError, RuntimeError, ValueError) as exc:
            raise EvalUnavailable(safe_reason(exc, provider=self._judge.provider)) from None
        evaluation = parse_evaluation(response.text, model=self._judge.model, base_first=base_first, secrets=secrets)
        self._evaluation = evaluation
        self._base_first = not base_first
        return evaluation

    async def reset(self) -> None:
        await self.prima.reset()
        self.comparison_id = f"compare-{uuid4().hex[:12]}"
        self.base_session_id = f"{self.comparison_id}-base"
        self.base_history.clear()
        self.turn_index = 0
        self.needs_reset = False
        self.latest_turn = None
        self._eval_transcript.clear()
        self._evaluation = None

    async def close(self) -> None:
        await self.prima.close()
