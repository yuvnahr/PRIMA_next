"""End-to-end runtime pipeline tests."""

from __future__ import annotations

from reflection.reflection_engine import ReflectionEngine
from runtime import PrimaRuntime, RuntimeResult


def test_runtime_executes_without_exceptions_and_returns_result() -> None:
    runtime = PrimaRuntime()

    result = runtime.process("I am nervous about tomorrow's exam")

    assert isinstance(result, RuntimeResult)
    assert result.final_response
    assert result.errors == ()


def test_runtime_updates_affect_retrieval_planning_and_memory() -> None:
    runtime = PrimaRuntime()

    result = runtime.process("The first sprout finally appeared after weeks of waiting.")

    assert result.affect_state["dominant_emotion"]
    assert isinstance(result.retrieved_memories, tuple)
    assert len(result.memory_notes_created) == 1
    assert result.confidence_score >= 0.0


def test_reflection_pipeline_executes_when_triggered() -> None:
    runtime = PrimaRuntime(reflection_engine=ReflectionEngine(trigger_threshold=0.0))

    result = runtime.process("I am unsure whether this plan has enough evidence.")

    assert result.reflection_triggered
