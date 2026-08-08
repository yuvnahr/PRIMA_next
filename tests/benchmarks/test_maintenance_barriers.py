from __future__ import annotations

from benchmarks.common.agent import BenchmarkAgent
from benchmarks.common.interfaces import AgentResponse, Conversation, ConversationQuestion, ConversationTurn
from benchmarks.common.runner import GenericBenchmarkRunner


class _BarrierAgent(BenchmarkAgent):
    def __init__(self) -> None:
        self.barriers: list[str] = []
        self.closed = False

    def reset(self) -> None:
        pass

    def process_turn(self, _turn: ConversationTurn) -> AgentResponse:
        return AgentResponse("")

    def answer_question(self, _question: ConversationQuestion) -> AgentResponse:
        return AgentResponse("answer")

    def get_state(self) -> dict[str, object]:
        return {"barriers": list(self.barriers)}

    def maintenance_barrier(self, barrier: str) -> bool:
        self.barriers.append(barrier)
        return True

    def close(self) -> None:
        assert self.barriers[-1] == "before_finalization"
        self.closed = True


def test_benchmark_runner_applies_barriers_before_questions_and_finalization() -> None:
    agent = _BarrierAgent()
    conversation = Conversation(
        id="conversation-1",
        turns=(ConversationTurn("User", "Remember Ada."),),
        questions=(ConversationQuestion("Who?", answer="Ada"),),
    )

    results = GenericBenchmarkRunner().run(agent, (conversation,))

    assert len(results) == 1
    assert agent.barriers == ["after_conversation", "before_question", "before_finalization"]
    assert agent.closed
