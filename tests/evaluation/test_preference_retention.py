"""Regression tests — Preference Retention metrics (Deliverable 14)."""

from __future__ import annotations

from evaluation.metrics.long_horizon_metrics import (
    preference_retention_at_turn,
    preference_retention_curve,
)
from evaluation.synthetic.user_generator import UserGenerator
from evaluation.synthetic.user_profile import Preference


def _record(turn: int, retrieved: str, ground_truth: str) -> dict[str, object]:
    return {
        "turn_index": turn,
        "retrieved_preference": retrieved,
        "ground_truth_preference": ground_truth,
    }


# ---------------------------------------------------------------------------
# Unit tests for metric functions
# ---------------------------------------------------------------------------

def test_preference_retention_all_correct() -> None:
    records = [
        _record(10, "VSCode", "VSCode"),
        _record(50, "VSCode", "VSCode"),
        _record(100, "Neovim", "Neovim"),
    ]
    score = preference_retention_at_turn(records, 100)
    assert abs(score - 1.0) < 1e-6


def test_preference_retention_all_wrong() -> None:
    records = [
        _record(10, "Emacs", "VSCode"),
        _record(50, "Emacs", "Neovim"),
    ]
    score = preference_retention_at_turn(records, 50)
    assert abs(score - 0.0) < 1e-6


def test_preference_retention_partial() -> None:
    records = [
        _record(10, "VSCode", "VSCode"),
        _record(50, "Emacs", "VSCode"),
        _record(100, "Neovim", "Neovim"),
        _record(100, "Emacs", "Neovim"),
    ]
    score = preference_retention_at_turn(records, 100)
    assert abs(score - 0.5) < 1e-6


def test_preference_retention_empty_gt_excluded() -> None:
    """Records without ground_truth are excluded from the denominator."""
    records = [
        _record(10, "VSCode", "VSCode"),
        {"turn_index": 20, "retrieved_preference": "anything", "ground_truth_preference": ""},
    ]
    score = preference_retention_at_turn(records, 50)
    # Only 1 valid record (turn 10), matched → 1.0
    assert abs(score - 1.0) < 1e-6


def test_preference_retention_empty_records() -> None:
    assert preference_retention_at_turn([], 100) == 0.0


def test_preference_retention_at_turn_filters_correctly() -> None:
    """Turns after the requested cutoff must be excluded."""
    records = [
        _record(10, "VSCode", "VSCode"),
        _record(200, "Emacs", "Neovim"),  # after cutoff of 100
    ]
    score = preference_retention_at_turn(records, 100)
    # Only turn 10 counts → 1.0
    assert abs(score - 1.0) < 1e-6


def test_preference_retention_curve_keys() -> None:
    records = [_record(t, "VSCode", "VSCode") for t in [10, 50, 100, 250, 500]]
    checkpoints = (10, 50, 100, 250, 500)
    curve = preference_retention_curve(records, checkpoints=checkpoints)
    assert set(curve.keys()) == {10, 50, 100, 250, 500}


def test_preference_retention_curve_values_in_range() -> None:
    records = [_record(t, "VSCode", "VSCode") for t in range(1, 501)]
    curve = preference_retention_curve(records)
    for val in curve.values():
        assert 0.0 <= val <= 1.0, f"Retention value {val} out of [0, 1]"


# ---------------------------------------------------------------------------
# Integration test: evolving preferences in synthetic user
# ---------------------------------------------------------------------------

def test_preference_evolution_in_synthetic_user() -> None:
    """Synthetic users must have at least two editor preference epochs."""
    gen = UserGenerator(n_users=10, min_turns=300, max_turns=400, seed_base=42)
    users = gen.generate_all()
    multi_editor_count = 0
    for user in users:
        editors = [p for p in user.preferences if p.category == "editor"]
        if len(editors) >= 2:
            multi_editor_count += 1
    assert multi_editor_count >= 5, (
        "At least half of users must have multi-epoch editor preferences"
    )


def test_preference_active_at_turn() -> None:
    """Preference.is_active correctly gates on turn_set and superseded_at_turn."""
    pref1 = Preference(category="editor", value="VSCode", turn_set=0, superseded_at_turn=100)
    pref2 = Preference(category="editor", value="Neovim", turn_set=100, superseded_at_turn=-1)

    assert pref1.is_active(50) is True
    assert pref1.is_active(99) is True
    assert pref1.is_active(100) is False  # superseded_at is exclusive end

    assert pref2.is_active(100) is True
    assert pref2.is_active(999) is True


def test_active_preferences_at_milestone_turns() -> None:
    """UserProfile.active_preferences returns only preferences active at the given turn."""
    gen = UserGenerator(n_users=5, min_turns=300, max_turns=400, seed_base=42)
    users = gen.generate_all()
    for user in users:
        prefs_early = user.profile.active_preferences(10)
        prefs_late = user.profile.active_preferences(350)
        # At least one editor preference at any time
        assert any(p.category == "editor" for p in prefs_early), (
            f"No editor preference at turn 10 for {user.user_id}"
        )
        assert any(p.category == "editor" for p in prefs_late), (
            f"No editor preference at turn 350 for {user.user_id}"
        )


def test_preference_retention_curve_improves_with_repetition() -> None:
    """When the same preference is consistently retrieved, retention should be 1.0 at all cps."""
    records = [_record(t, "Python", "Python") for t in range(1, 501)]
    curve = preference_retention_curve(records, checkpoints=(10, 50, 100, 250, 500))
    for cp, val in curve.items():
        assert abs(val - 1.0) < 1e-6, f"Expected 1.0 at cp={cp}, got {val}"
