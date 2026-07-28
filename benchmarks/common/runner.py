"""Benchmark-agnostic runner implementation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path

from benchmarks.common.agent import BenchmarkAgent
from benchmarks.common.interfaces import (
    BenchmarkResult,
    BenchmarkRunner,
    Conversation,
    ConversationQuestion,
)
from benchmarks.common.utils import configure_benchmark_logger


class GenericBenchmarkRunner(BenchmarkRunner):
    """Replay conversations through any compatible benchmark agent."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def run(
        self,
        agent: BenchmarkAgent,
        conversations: Iterable[Conversation],
        on_question_completed: Callable[[BenchmarkResult], None] | None = None,
    ) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []

        try:
            for conversation in conversations:
                self.logger.info("Conversation %s replay starting", conversation.id)
                agent.reset()
                for turn_index, turn in enumerate(conversation.turns, start=1):
                    agent.process_turn(turn)
                    self.logger.info("Conversation %s turn %s processed", conversation.id, turn_index)

                self.logger.info("Conversation %s replay complete", conversation.id)
                if conversation.questions:
                    for question_index, question in enumerate(conversation.questions, start=1):
                        response = agent.answer_question(ConversationQuestion(question=question.question, question_id=question.question_id, category=question.category))
                        self.logger.info("Conversation %s question %s answered", conversation.id, question_index)
                        result = BenchmarkResult(
                            conversation_id=conversation.id,
                            question_id=question.question_id,
                            prompt=question.question,
                            response=response,
                            expected_answer=question.answer,
                            metadata={
                                "category": question.category,
                                "evidence": list(question.evidence),
                                "conversation_metadata": dict(conversation.metadata),
                                "question_metadata": dict(question.metadata),
                                "agent_state": agent.get_state(),
                            },
                        )
                        results.append(result)
                        if on_question_completed is not None:
                            on_question_completed(result)
                    continue
                self.logger.info("Conversation %s has no questions; no result emitted", conversation.id)
        finally:
            agent.close()

        return results


class LoggingBenchmarkRunner(GenericBenchmarkRunner):
    """Generic runner configured with a dedicated benchmark log file."""

    def __init__(self, name: str, log_dir: str | Path) -> None:
        super().__init__(configure_benchmark_logger(name, Path(log_dir)))
