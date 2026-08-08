"""Generic benchmark agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from benchmarks.common.interfaces import AgentResponse, ConversationQuestion, ConversationTurn


class BenchmarkAgent(ABC):
    """Abstract interface used by benchmark runners to talk to agent runtimes."""

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state before replaying a new benchmark conversation."""

    @abstractmethod
    def process_turn(self, turn: ConversationTurn) -> AgentResponse:
        """Process one conversation turn during replay."""

    @abstractmethod
    def answer_question(self, question: ConversationQuestion) -> AgentResponse:
        """Answer a benchmark question after replay is complete."""

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return a serializable snapshot of the agent bridge state."""

    def maintenance_barrier(self, barrier: str) -> bool:
        """Apply an optional runtime maintenance barrier."""

        return False

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the agent bridge."""
