"""Acceptance policy for reflected prediction candidates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from affect.emotion_classifier import CANONICAL_EMOTION_ALIASES, FALLBACK_LEXICON

Outcome = str

_LABEL_ALIASES = {
    **CANONICAL_EMOTION_ALIASES,
    "love": "joy",
    "optimism": "anticipation",
    "gratitude": "trust",
    "approval": "trust",
    "caring": "trust",
    "excitement": "joy",
    "grief": "sadness",
    "nervousness": "fear",
    "annoyance": "anger",
    "disapproval": "disgust",
    "realization": "surprise",
    "curiosity": "surprise",
}


_GENERIC_TRUST_TERMS = {"friend"}
_INTENSE_TERMS = {
    "astonished",
    "awe",
    "fantastic",
    "gagged",
    "gross",
    "lonely",
    "overjoyed",
    "startled",
    "wonder",
    "wonderstruck",
    "yelled",
}


@dataclass(frozen=True, slots=True)
class ReflectionAcceptanceConfig:
    """Thresholds for deciding whether reflection may replace a prediction."""

    high_confidence_threshold: float = 0.75
    top_emotion_margin_threshold: float = 0.30
    trigger_confidence_ceiling: float = 0.10
    trigger_margin_ceiling: float = 0.10
    confidence_margin: float = 0.05


@dataclass(frozen=True, slots=True)
class ReflectionAcceptanceDecision:
    """Structured result of reflection candidate evaluation."""

    reflection_triggered: bool
    reflection_accepted: bool
    selected_prediction: str
    candidate_prediction: str
    original_prediction: str
    original_confidence: float
    candidate_confidence: float
    original_evidence_score: float
    reflection_evidence_score: float
    top_emotion_margin: float
    suppression_reason: str | None = None
    acceptance_reason: str | None = None


def top_emotion_margin(emotions: Mapping[str, float]) -> float:
    """Return the margin between the top two emotion scores."""
    scores = sorted((float(score) for score in emotions.values() if float(score) > 0.0), reverse=True)
    if not scores:
        return 0.0
    if len(scores) == 1:
        return round(scores[0], 6)
    return round(scores[0] - scores[1], 6)


def emotion_evidence_score(query: str, emotion: str) -> float:
    """Score direct lexical evidence for a canonical emotion label."""
    normalized_label = canonical_label(emotion)
    text = query.lower()
    score = 0.0
    for raw_label, phrases in FALLBACK_LEXICON.items():
        label = canonical_label(CANONICAL_EMOTION_ALIASES.get(raw_label, raw_label))
        if label != normalized_label:
            continue
        for phrase in phrases:
            if not _phrase_present(text, phrase):
                continue
            score += _phrase_weight(phrase)
    return round(score, 6)


def evaluate_reflection_acceptance(
    *,
    query: str,
    original_prediction: str,
    candidate_prediction: str,
    prediction_confidence: float,
    emotions: Mapping[str, float],
    original_confidence: float,
    candidate_confidence: float,
    config: ReflectionAcceptanceConfig | None = None,
) -> ReflectionAcceptanceDecision:
    """Apply guardrails and do-no-harm acceptance to a reflection candidate."""
    policy = config or ReflectionAcceptanceConfig()
    original = canonical_label(original_prediction)
    candidate = canonical_label(candidate_prediction)
    margin = top_emotion_margin(emotions)
    original_evidence = emotion_evidence_score(query, original)
    candidate_evidence = emotion_evidence_score(query, candidate)

    if original == candidate:
        return _retain(
            original,
            candidate,
            original_confidence,
            candidate_confidence,
            original_evidence,
            candidate_evidence,
            margin,
            triggered=False,
            reason="candidate_matches_original",
        )

    if prediction_confidence >= policy.high_confidence_threshold and margin >= policy.top_emotion_margin_threshold:
        return _retain(
            original,
            candidate,
            original_confidence,
            candidate_confidence,
            original_evidence,
            candidate_evidence,
            margin,
            triggered=False,
            reason="high_confidence_clear_margin",
        )

    ambiguous = prediction_confidence <= policy.trigger_confidence_ceiling or margin <= policy.trigger_margin_ceiling
    evidence_warrants_candidate = candidate_evidence >= original_evidence
    if not ambiguous or not evidence_warrants_candidate:
        return _retain(
            original,
            candidate,
            original_confidence,
            candidate_confidence,
            original_evidence,
            candidate_evidence,
            margin,
            triggered=False,
            reason="insufficient_reflection_trigger_evidence",
        )

    confidence_improved = candidate_confidence > original_confidence + policy.confidence_margin
    evidence_improved = candidate_evidence > original_evidence
    accepted = confidence_improved and evidence_improved
    return ReflectionAcceptanceDecision(
        reflection_triggered=True,
        reflection_accepted=accepted,
        selected_prediction=candidate if accepted else original,
        candidate_prediction=candidate,
        original_prediction=original,
        original_confidence=round(float(original_confidence), 6),
        candidate_confidence=round(float(candidate_confidence), 6),
        original_evidence_score=original_evidence,
        reflection_evidence_score=candidate_evidence,
        top_emotion_margin=margin,
        suppression_reason=None if accepted else "do_no_harm_policy",
        acceptance_reason="confidence_and_evidence_improved" if accepted else None,
    )


def classify_reflection_outcome(
    before_prediction: str,
    after_prediction: str,
    ground_truth: str,
    *,
    reflection_triggered: bool,
) -> Outcome:
    """Classify benchmark impact for a reflection event."""
    if not reflection_triggered:
        return "NEUTRAL"
    before_correct = canonical_label(before_prediction) == canonical_label(ground_truth)
    after_correct = canonical_label(after_prediction) == canonical_label(ground_truth)
    if not before_correct and after_correct:
        return "IMPROVEMENT"
    if before_correct and not after_correct:
        return "HARM"
    return "NEUTRAL"


def reflection_recovery_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize reflection recovery and acceptance metrics."""
    total = len(records)
    triggered = sum(1 for record in records if record.get("reflection_triggered"))
    accepted = sum(1 for record in records if record.get("reflection_accepted"))
    improvements = sum(1 for record in records if record.get("reflection_outcome") == "IMPROVEMENT")
    harms = sum(1 for record in records if record.get("reflection_outcome") == "HARM")
    return {
        "total_samples": total,
        "reflection_trigger_count": triggered,
        "reflection_acceptance_count": accepted,
        "improvement_count": improvements,
        "harm_count": harms,
        "reflection_trigger_rate": _rate(triggered, total),
        "reflection_acceptance_rate": _rate(accepted, triggered),
        "improvement_rate": _rate(improvements, total),
        "harm_rate": _rate(harms, total),
        "net_gain": round(_rate(improvements, total) - _rate(harms, total), 6),
    }


def _retain(
    original: str,
    candidate: str,
    original_confidence: float,
    candidate_confidence: float,
    original_evidence: float,
    candidate_evidence: float,
    margin: float,
    *,
    triggered: bool,
    reason: str,
) -> ReflectionAcceptanceDecision:
    return ReflectionAcceptanceDecision(
        reflection_triggered=triggered,
        reflection_accepted=False,
        selected_prediction=original,
        candidate_prediction=candidate,
        original_prediction=original,
        original_confidence=round(float(original_confidence), 6),
        candidate_confidence=round(float(candidate_confidence), 6),
        original_evidence_score=original_evidence,
        reflection_evidence_score=candidate_evidence,
        top_emotion_margin=margin,
        suppression_reason=reason,
        acceptance_reason=None,
    )


def _phrase_present(text: str, phrase: str) -> bool:
    normalized = phrase.lower()
    if " " in normalized:
        return normalized in text
    return re.search(rf"\b{re.escape(normalized)}\b", text) is not None


def _phrase_weight(phrase: str) -> float:
    normalized = phrase.lower()
    if normalized in _GENERIC_TRUST_TERMS:
        return 0.15
    if normalized in _INTENSE_TERMS:
        return 1.5
    return 1.0


def canonical_label(label: str) -> str:
    return _LABEL_ALIASES.get(str(label).strip().lower(), str(label).strip().lower())


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0
