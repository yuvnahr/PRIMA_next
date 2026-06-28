"""Regression tests — Identity Consistency metrics (Deliverable 14)."""

from __future__ import annotations

from evaluation.metrics.long_horizon_metrics import (
    identity_consistency_curve,
    identity_consistency_score,
)


def _make_probe(turn: int, field: str, ground_truth: str, consistent: bool) -> dict[str, object]:
    return {
        "probe_turn": turn,
        "field": field,
        "ground_truth": ground_truth,
        "retrieved_value": ground_truth if consistent else "wrong",
        "consistent": consistent,
    }


def test_identity_consistency_perfect() -> None:
    probes = [
        _make_probe(10, "name", "Alice Chen", True),
        _make_probe(10, "occupation", "Software Engineer", True),
        _make_probe(50, "workplace", "DeepMind", True),
    ]
    score = identity_consistency_score(probes)
    assert score == 1.0, "All consistent probes must give score of 1.0"


def test_identity_consistency_zero() -> None:
    probes = [
        _make_probe(10, "name", "Alice Chen", False),
        _make_probe(50, "occupation", "Software Engineer", False),
    ]
    score = identity_consistency_score(probes)
    assert score == 0.0, "All inconsistent probes must give score of 0.0"


def test_identity_consistency_partial() -> None:
    probes = [
        _make_probe(10, "name", "Alice Chen", True),
        _make_probe(50, "occupation", "Software Engineer", False),
        _make_probe(100, "workplace", "DeepMind", True),
        _make_probe(100, "goal", "lead a team", False),
    ]
    score = identity_consistency_score(probes)
    assert abs(score - 0.5) < 1e-6, f"Expected 0.5, got {score}"


def test_identity_consistency_empty() -> None:
    assert identity_consistency_score([]) == 0.0


def test_identity_consistency_curve_keys() -> None:
    probes = [_make_probe(t, "name", "Alex", True) for t in [10, 50, 100, 250, 500]]
    curve = identity_consistency_curve(probes, checkpoints=(10, 50, 100, 250, 500))
    assert set(curve.keys()) == {10, 50, 100, 250, 500}


def test_identity_consistency_curve_monotone() -> None:
    """More turns passed → curve should accumulate more consistent probes."""
    probes = [_make_probe(t, "name", "Alex", True) for t in [10, 50, 100]]
    curve = identity_consistency_curve(probes, checkpoints=(10, 50, 100))
    # All consistent, so all checkpoints should be 1.0
    for cp, val in curve.items():
        assert abs(val - 1.0) < 1e-6, f"Checkpoint {cp} should be 1.0, got {val}"


def test_identity_consistency_curve_increases_with_data() -> None:
    """Early probes inconsistent, later probes consistent — score at later cp ≥ earlier."""
    probes = [
        _make_probe(10, "name", "Alex", False),
        _make_probe(50, "name", "Alex", True),
        _make_probe(100, "name", "Alex", True),
    ]
    curve = identity_consistency_curve(probes, checkpoints=(10, 50, 100))
    # At cp=10: only 1 probe, inconsistent → 0.0
    assert abs(curve[10] - 0.0) < 1e-6
    # At cp=50: 2 probes, 1 consistent → 0.5
    assert abs(curve[50] - 0.5) < 1e-6
    # At cp=100: 3 probes, 2 consistent → 0.666...
    assert curve[100] > curve[50]


def test_identity_fields_probed() -> None:
    """Probes cover all four key identity fields."""
    fields = ["name", "occupation", "workplace", "goal"]
    probes = [_make_probe(10, f, "v", True) for f in fields]
    score = identity_consistency_score(probes)
    assert score == 1.0


def test_multi_field_consistency() -> None:
    """A single turn can probe multiple fields simultaneously."""
    probes = [
        _make_probe(100, "name", "Jordan Patel", True),
        _make_probe(100, "occupation", "Data Scientist", True),
        _make_probe(100, "workplace", "Google Brain", False),
        _make_probe(100, "goal", "publish research", True),
    ]
    score = identity_consistency_score(probes)
    assert abs(score - 0.75) < 1e-6
