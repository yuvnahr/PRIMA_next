"""Benchmark-agnostic runner implementation."""

from __future__ import annotations

from collections.abc import Iterable

from benchmarks.common.interfaces import AgentRuntime, BenchmarkRunner, Conversation, RunnerResult


class GenericBenchmarkRunner(BenchmarkRunner):
    """Run any compatible agent runtime over normalized conversations."""

    def run(self, agent: AgentRuntime, conversations: Iterable[Conversation]) -> list[RunnerResult]:
        results: list[RunnerResult] = []

        for conversation in conversations:
            if conversation.questions:
                for question in conversation.questions:
                    response = agent.respond(conversation, question)
                    results.append(
                        RunnerResult(
                            conversation_id=conversation.id,
                            question_id=question.question_id,
                            prompt=question.question,
                            response=response,
                            expected_answer=question.answer,
                            metadata={"category": question.category},
                        )
                    )
                continue

            response = agent.respond(conversation)
            results.append(
                RunnerResult(
                    conversation_id=conversation.id,
                    question_id=None,
                    prompt="",
                    response=response,
                )
            )

        return results
