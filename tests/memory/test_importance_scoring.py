"""Tests for memory importance score aggregation."""

from __future__ import annotations

from types import SimpleNamespace

from memory.maintenance.importance_types import ImportanceWeights, MemoryImportanceConfig
from memory.maintenance.memory_importance import MemoryImportanceEngine
from memory.memory_repository import InMemoryMemoryRepository


def test_importance_score_aggregates_configured_weights() -> None:
    repository = InMemoryMemoryRepository()
    engine = MemoryImportanceEngine(
        repository,
        config=MemoryImportanceConfig(
            threshold=0.55,
            weights=ImportanceWeights(
                novelty=1.0,
                emotional_salience=0.0,
                user_relevance=0.0,
                recurrence=0.0,
                reflection=0.0,
            ),
        ),
    )

    score = engine.score("My long-term career goal is to study robotics.")

    assert score.novelty_score == 1.0
    assert score.total_score == 1.0


def test_emotional_salience_uses_affect_output() -> None:
    profile = SimpleNamespace(dominant_emotion="fear", confidence=0.9)
    pad_state = SimpleNamespace(arousal=0.8)
    affect_update = SimpleNamespace(profile=profile, pad_state=pad_state, salience_score=0.9)

    score = MemoryImportanceEngine(InMemoryMemoryRepository()).emotional_salience_score(
        "I was terrified by the sudden alarm.",
        affect_update,
    )

    assert score > 0.75
