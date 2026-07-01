"""Tests for reflection acceptance guardrails."""

from __future__ import annotations

from reflection.reflection_acceptance import (
    classify_reflection_outcome,
    evaluate_reflection_acceptance,
    reflection_recovery_summary,
)


def test_high_confidence_clear_margin_suppresses_reflection() -> None:
    decision = evaluate_reflection_acceptance(
        query="I felt pure joy all afternoon.",
        original_prediction="joy",
        candidate_prediction="trust",
        prediction_confidence=0.82,
        emotions={"joy": 0.82, "trust": 0.11},
        original_confidence=0.70,
        candidate_confidence=0.90,
    )

    assert decision.reflection_triggered is False
    assert decision.reflection_accepted is False
    assert decision.selected_prediction == "joy"
    assert decision.suppression_reason == "high_confidence_clear_margin"


def test_do_no_harm_accepts_only_confidence_and_evidence_gain() -> None:
    decision = evaluate_reflection_acceptance(
        query="My closest friend felt overjoyed by the surprise party.",
        original_prediction="trust",
        candidate_prediction="joy",
        prediction_confidence=0.0,
        emotions={"trust": 0.333, "joy": 0.333, "surprise": 0.333},
        original_confidence=0.50,
        candidate_confidence=0.65,
    )

    assert decision.reflection_triggered is True
    assert decision.reflection_accepted is True
    assert decision.selected_prediction == "joy"
    assert decision.reflection_evidence_score > decision.original_evidence_score


def test_do_no_harm_rejects_candidate_without_evidence_gain() -> None:
    decision = evaluate_reflection_acceptance(
        query="My closest friend felt unsafe after hearing footsteps behind them.",
        original_prediction="fear",
        candidate_prediction="trust",
        prediction_confidence=0.0,
        emotions={"fear": 0.5, "trust": 0.5},
        original_confidence=0.50,
        candidate_confidence=0.65,
    )

    assert decision.reflection_triggered is False
    assert decision.reflection_accepted is False
    assert decision.selected_prediction == "fear"


def test_outcome_and_recovery_summary() -> None:
    records = [
        {"reflection_triggered": True, "reflection_accepted": True, "reflection_outcome": "IMPROVEMENT"},
        {"reflection_triggered": True, "reflection_accepted": False, "reflection_outcome": "NEUTRAL"},
        {"reflection_triggered": False, "reflection_accepted": False, "reflection_outcome": "NEUTRAL"},
    ]

    assert classify_reflection_outcome("trust", "joy", "joy", reflection_triggered=True) == "IMPROVEMENT"
    assert classify_reflection_outcome("joy", "trust", "joy", reflection_triggered=True) == "HARM"
    assert reflection_recovery_summary(records) == {
        "total_samples": 3,
        "reflection_trigger_count": 2,
        "reflection_acceptance_count": 1,
        "improvement_count": 1,
        "harm_count": 0,
        "reflection_trigger_rate": 0.666667,
        "reflection_acceptance_rate": 0.5,
        "improvement_rate": 0.333333,
        "harm_rate": 0.0,
        "net_gain": 0.333333,
    }
