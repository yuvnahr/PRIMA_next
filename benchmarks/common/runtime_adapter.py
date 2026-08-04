"""Runtime adapter that exposes PRIMA through the benchmark agent interface."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmarks.common.agent import BenchmarkAgent
from benchmarks.common.interfaces import AgentResponse, ConversationQuestion, ConversationTurn
from config.runtime_mode import RuntimeMode
from runtime.prima_runtime import PrimaRuntime
from runtime.runtime_context import RuntimeContext


class PrimaRuntimeAdapter(BenchmarkAgent):
    """Bridge benchmark conversations into the black-box PRIMA runtime."""

    def __init__(
        self,
        runtime_factory: Callable[..., PrimaRuntime] = PrimaRuntime,
        log_path: str | Path | None = None,
        document_ingestion: bool = False,
    ) -> None:
        self.runtime_factory = runtime_factory
        self.log_path = Path(log_path or "logs/prima_runtime_adapter.log")
        self.runtime: PrimaRuntime | None = None
        self.context: RuntimeContext | None = None
        self.document_ingestion = document_ingestion
        self.answer_options: dict[str, Any] = {}
        self.turn_count = 0
        self.question_count = 0
        self.ingestion_time = 0.0
        self.reasoning_time = 0.0
        self.reset()

    def reset(self) -> None:
        """Create a fresh PRIMA runtime and context for one benchmark conversation."""

        self.runtime = self.runtime_factory(
            log_path=self.log_path,
            mode=RuntimeMode.BENCHMARK,
            memory_backend="in_memory",
        )
        self.context = RuntimeContext()
        self.turn_count = 0
        self.question_count = 0
        self.ingestion_time = 0.0
        self.reasoning_time = 0.0

    def process_turn(self, turn: ConversationTurn) -> AgentResponse:
        """Replay a benchmark conversation turn through PRIMA."""

        runtime = self._runtime()
        context = self._context()
        context.session_id = turn.session_id or context.session_id
        context.turn_id = turn.turn_id or context.turn_id
        if turn.timestamp:
            context.timestamp = datetime.strptime(turn.timestamp, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)

        started = perf_counter()
        result = (
            runtime.ingest_document(self._format_turn(turn), metadata=turn.metadata)
            if self.document_ingestion
            else runtime.process(self._format_turn(turn), context=context)
        )
        self.ingestion_time += perf_counter() - started
        self.turn_count += 1
        return self._response_from_runtime(result)

    def answer_question(self, question: ConversationQuestion) -> AgentResponse:
        """Ask a benchmark question after replay has completed."""

        started = perf_counter()
        result = self._runtime().answer_question(
            self._format_question(question),
            context=self._context(),
            **self.answer_options,
        )
        self.reasoning_time += perf_counter() - started
        self.question_count += 1
        return self._response_from_runtime(result)

    def get_state(self) -> dict[str, Any]:
        """Return bridge-level state without exposing PRIMA internals."""

        context = self.context.to_dict() if self.context is not None else {}
        return {
            "turn_count": self.turn_count,
            "question_count": self.question_count,
            "ingestion_time": self.ingestion_time,
            "reasoning_time": self.reasoning_time,
            "context": context,
            "runtime_manifest": self.runtime.runtime_manifest()
            if self.runtime is not None and hasattr(self.runtime, "runtime_manifest")
            else {"schema_version": "1.0", "runtime_mode": "benchmark", "memory_repository": "in_memory"},
        }

    def close(self) -> None:
        """Release references to the active runtime and context."""

        self.runtime = None
        self.context = None

    def _runtime(self) -> PrimaRuntime:
        if self.runtime is None:
            self.reset()
        if self.runtime is None:
            raise RuntimeError("PRIMA runtime adapter failed to initialize a runtime.")
        return self.runtime

    def _context(self) -> RuntimeContext:
        if self.context is None:
            self.context = RuntimeContext()
        return self.context

    def _format_turn(self, turn: ConversationTurn) -> str:
        speaker = turn.speaker.strip() or "Speaker"
        return f"{speaker}: {turn.text}"

    def _format_question(self, question: ConversationQuestion) -> str:
        return question.question

    def _response_from_runtime(self, result: Any) -> AgentResponse:
        metadata = result.to_dict() if hasattr(result, "to_dict") else {"raw_result": str(result)}
        text = str(getattr(result, "final_response", ""))
        return AgentResponse(text=text, metadata=metadata)
