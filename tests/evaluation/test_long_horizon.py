"""Regression tests — Long-Horizon runner smoke tests (Deliverable 14)."""

from __future__ import annotations

from evaluation.metrics.long_horizon_metrics import (
    build_long_horizon_summary,
    emotion_continuity_score,
    fact_retention_curve,
    forgetting_evaluation,
    reflection_stability,
    retrieval_stability,
)
from evaluation.runners.long_horizon_runner import (
    LongHorizonRunner,
    _check_identity_field,
    _check_preference_retrieved,
    _get_dominant_emotion,
)

# ---------------------------------------------------------------------------
# Helper probe utilities
# ---------------------------------------------------------------------------

def test_check_identity_field_found() -> None:
    """Heuristic probe returns True when expected value appears in any retrieved memory."""

    class _FakeNote:
        content = "My name is Alice Chen and I work at DeepMind."

    class _FakeItem:
        note = _FakeNote()

    assert _check_identity_field((_FakeItem(),), "Alice Chen") is True
    assert _check_identity_field((_FakeItem(),), "DeepMind") is True


def test_check_identity_field_not_found() -> None:
    class _FakeNote:
        content = "I enjoy rock climbing."

    class _FakeItem:
        note = _FakeNote()

    assert _check_identity_field((_FakeItem(),), "Alice Chen") is False


def test_check_identity_field_empty_value() -> None:
    assert _check_identity_field((), "") is False


def test_check_preference_retrieved_match() -> None:
    class _FakeNote:
        content = "I prefer using VSCode for all my development work."

    class _FakeItem:
        note = _FakeNote()

    result = _check_preference_retrieved((_FakeItem(),), "VSCode")
    assert result == "VSCode"


def test_check_preference_retrieved_no_match() -> None:
    class _FakeNote:
        content = "I enjoy hiking on weekends."

    class _FakeItem:
        note = _FakeNote()

    result = _check_preference_retrieved((_FakeItem(),), "Neovim")
    assert result == ""


def test_get_dominant_emotion() -> None:
    assert _get_dominant_emotion({"dominant_emotion": "joy"}) == "joy"
    assert _get_dominant_emotion({}) == "neutral"
    assert _get_dominant_emotion({"dominant_emotion": "fear"}) == "fear"


# ---------------------------------------------------------------------------
# Metric function integration
# ---------------------------------------------------------------------------

def test_retrieval_stability_equal_distribution() -> None:
    records = [
        {"turn_index": t, "retrieval_count": 3}
        for t in range(30)
    ]
    result = retrieval_stability(records)
    # All thirds have equal retrieval → degradation ratio ~1.0
    assert abs(result["degradation_ratio"] - 1.0) < 0.1


def test_retrieval_stability_empty() -> None:
    result = retrieval_stability([])
    assert result["degradation_ratio"] == 1.0


def test_reflection_stability_no_reflection() -> None:
    records = [
        {"turn_index": t, "reflection_triggered": False, "correction_count": 0,
         "reflection_utility_score": 0.0}
        for t in range(20)
    ]
    result = reflection_stability(records)
    assert result["reflection_frequency"] == 0.0
    assert result["correction_frequency"] == 0.0


def test_reflection_stability_all_reflection() -> None:
    records = [
        {"turn_index": t, "reflection_triggered": True, "correction_count": 1,
         "reflection_utility_score": 0.5}
        for t in range(20)
    ]
    result = reflection_stability(records)
    assert abs(result["reflection_frequency"] - 1.0) < 1e-6
    assert abs(result["correction_frequency"] - 1.0) < 1e-6
    assert abs(result["mean_utility"] - 0.5) < 1e-6


def test_forgetting_evaluation_no_forgotten() -> None:
    result = forgetting_evaluation([], [])
    assert result["total_forgotten"] == 0
    assert result["appropriate_forgetting_rate"] == 1.0


def test_emotion_continuity_all_match() -> None:
    records = [
        {"retrieved_emotion": "joy", "expected_emotion": "joy"},
        {"retrieved_emotion": "trust", "expected_emotion": "trust"},
    ]
    assert emotion_continuity_score(records) == 1.0


def test_fact_retention_curve_at_zero_turns() -> None:
    curve = fact_retention_curve([], checkpoints=(10, 50))
    for val in curve.values():
        assert val == 0.0


def test_build_long_horizon_summary_empty() -> None:
    result = build_long_horizon_summary([])
    assert result == {}


def test_build_long_horizon_summary_single_user() -> None:
    user_result = {
        "user_id": "user_0000",
        "preference_retention": 0.95,
        "preference_retention_curve": {"10": 0.9, "50": 0.95},
        "identity_consistency": 0.92,
        "identity_consistency_curve": {"10": 0.8, "50": 0.92},
        "emotion_continuity": 0.8,
        "emotion_continuity_curve": {"10": 0.75},
        "fact_retention_curve": {"10": 0.88},
        "retrieval_hit_rate": 0.7,
        "mean_reflection_utility": 0.4,
        "total_memories_created": 120,
        "average_latency_ms": 15.3,
        "retrieval_degradation_ratio": 1.02,
    }
    summary = build_long_horizon_summary([user_result])
    assert abs(summary["preference_retention"] - 0.95) < 1e-6
    assert abs(summary["identity_consistency"] - 0.92) < 1e-6
    assert summary["total_users"] == 1


# ---------------------------------------------------------------------------
# End-to-end smoke test (tiny cohort)
# ---------------------------------------------------------------------------

def test_long_horizon_runner_smoke(tmp_path: object) -> None:
    """Runner completes on a tiny 2-user, 30-turn cohort without errors."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = LongHorizonRunner(
            n_users=2,
            min_turns=30,
            max_turns=40,
            seed_base=42,
            results_dir=Path(tmpdir),
        )
        output = runner.run()

    assert output["n_users"] == 2
    assert "summary" in output


def test_long_horizon_runner_output_files(tmp_path: object) -> None:
    """All required output files are created."""
    import tempfile
    from pathlib import Path

    expected_files = [
        "long_horizon_results.json",
        "long_horizon_trace.json",
        "long_horizon_summary.json",
        "memory_retention_analysis.json",
        "memory_evolution_analysis.json",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)
        runner = LongHorizonRunner(
            n_users=2,
            min_turns=20,
            max_turns=25,
            seed_base=42,
            results_dir=results_dir,
        )
        runner.run()

        for fname in expected_files:
            path = results_dir / fname
            assert path.exists(), f"Expected output file missing: {fname}"
            assert path.stat().st_size > 0, f"Output file is empty: {fname}"
