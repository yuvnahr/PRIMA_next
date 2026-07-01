"""Regression tests — Memory Growth metrics (Deliverable 14)."""

from __future__ import annotations

from evaluation.metrics.long_horizon_metrics import (
    average_latency,
    latency_vs_memory,
    memory_density_at_turn,
    memory_evolution_analysis,
    memory_growth_curve,
)
from evaluation.synthetic.user_generator import UserGenerator


def _record(turn: int, created: int, latency: float = 10.0) -> dict[str, object]:
    return {
        "turn_index": turn,
        "memory_notes_created": created,
        "latency_ms": latency,
    }


# ---------------------------------------------------------------------------
# memory_growth_curve
# ---------------------------------------------------------------------------

def test_memory_growth_curve_cumulative() -> None:
    records = [
        _record(0, 1),
        _record(1, 0),
        _record(2, 2),
        _record(3, 1),
    ]
    curve = memory_growth_curve(records)
    cumulative = [pt["cumulative_memories"] for pt in curve]
    assert cumulative == [1, 1, 3, 4], f"Unexpected cumulative: {cumulative}"


def test_memory_growth_curve_sorted_by_turn() -> None:
    records = [_record(5, 1), _record(2, 1), _record(0, 1)]
    curve = memory_growth_curve(records)
    turns = [pt["turn_index"] for pt in curve]
    assert turns == sorted(turns)


def test_memory_growth_curve_density() -> None:
    records = [_record(t, 1) for t in range(10)]
    curve = memory_growth_curve(records)
    for pt in curve:
        t = pt["turn_index"]
        expected_density = (t + 1) / (t + 1)  # 1.0 always when every turn stores
        assert abs(float(pt["memory_density"]) - expected_density) < 1e-3


def test_memory_growth_curve_empty() -> None:
    assert memory_growth_curve([]) == []


# ---------------------------------------------------------------------------
# memory_density_at_turn
# ---------------------------------------------------------------------------

def test_memory_density_at_turn() -> None:
    records = [_record(t, 1) for t in range(20)]
    curve = memory_growth_curve(records)
    density = memory_density_at_turn(curve, 9)
    assert density > 0.0


def test_memory_density_at_turn_zero() -> None:
    assert memory_density_at_turn([], 100) == 0.0


# ---------------------------------------------------------------------------
# latency_vs_memory
# ---------------------------------------------------------------------------

def test_latency_vs_memory_structure() -> None:
    records = [_record(t, 1, latency=float(t + 1)) for t in range(5)]
    lvm = latency_vs_memory(records)
    assert len(lvm) == 5
    for entry in lvm:
        assert "turn_index" in entry
        assert "latency_ms" in entry
        assert "cumulative_memories" in entry


def test_latency_vs_memory_cumulative() -> None:
    records = [_record(t, 1, latency=5.0) for t in range(3)]
    lvm = latency_vs_memory(records)
    cumulative = [pt["cumulative_memories"] for pt in lvm]
    assert cumulative == [1, 2, 3]


# ---------------------------------------------------------------------------
# average_latency
# ---------------------------------------------------------------------------

def test_average_latency_correct() -> None:
    records = [{"latency_ms": 10.0}, {"latency_ms": 20.0}, {"latency_ms": 30.0}]
    assert abs(average_latency(records) - 20.0) < 1e-6


def test_average_latency_empty() -> None:
    assert average_latency([]) == 0.0


# ---------------------------------------------------------------------------
# memory_evolution_analysis
# ---------------------------------------------------------------------------

def test_memory_evolution_analysis_empty() -> None:
    result = memory_evolution_analysis([])
    assert result["total_memories_created"] == 0


def test_memory_evolution_analysis_counts() -> None:
    records = [
        {"memory_notes_created": 1, "abstraction_count": 0, "semantic_merge_count": 0},
        {"memory_notes_created": 2, "abstraction_count": 1, "semantic_merge_count": 0},
        {"memory_notes_created": 0, "abstraction_count": 0, "semantic_merge_count": 1},
    ]
    result = memory_evolution_analysis(records)
    assert result["total_memories_created"] == 3
    assert result["abstraction_count"] == 1
    assert result["semantic_merge_count"] == 1


# ---------------------------------------------------------------------------
# Integration: generated users produce valid memory data
# ---------------------------------------------------------------------------

def test_synthetic_users_produce_memory_growth() -> None:
    """Conversation generator produces turn records that drive memory growth."""
    gen = UserGenerator(n_users=3, min_turns=50, max_turns=60, seed_base=42)
    users = gen.generate_all()
    for user in users:
        assert user.total_turns >= 50, f"{user.user_id} too few turns"
        # Each turn produces an utterance (the driver for memory admission)
        assert all(t.utterance for t in user.conversation_turns), (
            f"{user.user_id}: some turns have empty utterances"
        )


def test_memory_growth_curve_non_decreasing() -> None:
    """Cumulative memory count must never decrease across turns."""
    records = [_record(t, (t % 3 == 0)) for t in range(30)]
    curve = memory_growth_curve(records)
    cumulative = [pt["cumulative_memories"] for pt in curve]
    for i in range(1, len(cumulative)):
        assert cumulative[i] >= cumulative[i - 1], (
            f"Memory count decreased at index {i}: {cumulative[i - 1]} → {cumulative[i]}"
        )
