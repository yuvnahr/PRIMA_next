"""Benchmark-neutral contracts and conversation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarks.common.agent import BenchmarkAgent


Metadata = dict[str, Any]


@dataclass(frozen=True)
class ConversationTurn:
    """A single normalized turn in a benchmark conversation."""

    speaker: str
    text: str
    turn_id: str | None = None
    session_id: str | None = None
    timestamp: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationQuestion:
    """A normalized benchmark question associated with a conversation."""

    question: str
    answer: str | None = None
    question_id: str | None = None
    category: str | None = None
    evidence: Sequence[str] = field(default_factory=tuple)
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Conversation:
    """Benchmark-neutral representation consumed by agent runtimes."""

    id: str
    turns: Sequence[ConversationTurn]
    questions: Sequence[ConversationQuestion] = field(default_factory=tuple)
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResponse:
    """A generic response returned by an agent runtime."""

    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    """A generic output record produced by a benchmark runner."""

    conversation_id: str
    question_id: str | None
    prompt: str
    response: AgentResponse
    expected_answer: str | None = None
    metadata: Metadata = field(default_factory=dict)


RunnerResult = BenchmarkResult


class BenchmarkDataset(ABC):
    """Interface implemented by benchmark dataset integrations."""

    @abstractmethod
    def load(self) -> Any:
        """Load raw benchmark data."""

    @abstractmethod
    def conversations(self) -> Sequence[Conversation]:
        """Return normalized benchmark conversations."""


class BenchmarkRunner(ABC):
    """Interface for benchmark-agnostic runners."""

    @abstractmethod
    def run(self, agent: BenchmarkAgent, conversations: Iterable[Conversation]) -> Sequence[BenchmarkResult]:
        """Run an agent over normalized benchmark conversations."""


class BenchmarkEvaluator(ABC):
    """Interface for benchmark evaluators."""

    @abstractmethod
    def evaluate(self, results: Iterable[BenchmarkResult]) -> Mapping[str, Any]:
        """Compute benchmark metrics from runner outputs."""
