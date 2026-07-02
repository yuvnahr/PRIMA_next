"""Benchmark-neutral contracts and conversation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence


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
class RunnerResult:
    """A generic output record produced by a benchmark runner."""

    conversation_id: str
    question_id: str | None
    prompt: str
    response: AgentResponse
    expected_answer: str | None = None
    metadata: Metadata = field(default_factory=dict)


class AgentRuntime(Protocol):
    """Minimal protocol for any runtime that can answer benchmark prompts."""

    def respond(self, conversation: Conversation, question: ConversationQuestion | None = None) -> AgentResponse:
        """Return a response for a conversation, optionally scoped to a question."""


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
    def run(self, agent: AgentRuntime, conversations: Iterable[Conversation]) -> Sequence[RunnerResult]:
        """Run an agent over normalized benchmark conversations."""


class BenchmarkEvaluator(ABC):
    """Interface for benchmark evaluators."""

    @abstractmethod
    def evaluate(self, results: Iterable[RunnerResult]) -> Mapping[str, Any]:
        """Compute benchmark metrics from runner outputs."""
