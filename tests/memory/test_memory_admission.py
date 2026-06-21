"""Tests for memory admission decisions."""

from __future__ import annotations

from memory.maintenance.importance_types import ImportanceWeights, MemoryImportanceConfig
from memory.maintenance.memory_importance import MemoryImportanceEngine
from memory.memory_repository import InMemoryMemoryRepository


def test_memory_admission_stores_above_threshold() -> None:
    engine = MemoryImportanceEngine(
        InMemoryMemoryRepository(),
        config=MemoryImportanceConfig(
            threshold=0.55,
            weights=ImportanceWeights(novelty=1.0, emotional_salience=0.0, user_relevance=0.0, recurrence=0.0, reflection=0.0),
        ),
    )

    decision = engine.decide("Please remember my research interest in adaptive memory systems.")

    assert decision.stored is True
    assert decision.reason == "importance_threshold_met"


def test_memory_admission_rejects_below_threshold() -> None:
    engine = MemoryImportanceEngine(
        InMemoryMemoryRepository(),
        config=MemoryImportanceConfig(
            threshold=0.90,
            weights=ImportanceWeights(novelty=0.0, emotional_salience=0.0, user_relevance=0.0, recurrence=0.0, reflection=0.0),
        ),
    )

    decision = engine.decide("Nice weather today.")

    assert decision.stored is False
    assert decision.reason == "importance_below_threshold"
