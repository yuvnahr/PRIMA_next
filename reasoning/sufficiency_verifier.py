"""Structured, deterministic evidence sufficiency checks."""

from __future__ import annotations

import re

from reasoning.models import EvidenceState, SufficiencyDecision, SufficiencyStatus

_STOP_WORDS = {"a", "an", "and", "are", "be", "does", "for", "in", "is", "of", "the", "to", "what", "which", "who"}


def _terms(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-zA-Z]{3,}", text.lower()) if word not in _STOP_WORDS}


class SufficiencyVerifier:
    def evaluate(self, state: EvidenceState) -> SufficiencyDecision:
        if not state.evidence_items:
            return SufficiencyDecision(SufficiencyStatus.UNANSWERABLE, 0.0, False, reason_code="NO_EVIDENCE")
        evidence_text = " ".join(item.text for item in state.evidence_items)
        question_terms = _terms(state.request.question)
        coverage = len(question_terms & _terms(evidence_text)) / max(1, len(question_terms))
        conflicting = self._contradictions(state)
        if conflicting:
            return SufficiencyDecision(SufficiencyStatus.CONTRADICTORY, min(state.confidence, coverage), False,
                                       contradictions=tuple(conflicting), recommended_action="stop", reason_code="CONTRADICTORY_EVIDENCE")
        if coverage >= 0.60:
            return SufficiencyDecision(SufficiencyStatus.SUFFICIENT, max(state.confidence, coverage), True,
                                       recommended_action="synthesize", reason_code="QUESTION_TERMS_SUPPORTED")
        return SufficiencyDecision(SufficiencyStatus.NEED_MORE_EVIDENCE, max(0.0, coverage), False,
                                   missing_information=("Evidence does not yet cover the requested relation.",),
                                   recommended_action="retrieve", reason_code="MISSING_RELATION")

    def _contradictions(self, state: EvidenceState) -> list[str]:
        positives: set[str] = set()
        negatives: set[str] = set()
        for item in state.evidence_items:
            for subject, negated, value in re.findall(r"\b([A-Z][\w-]*(?:\s+[A-Z][\w-]*)*)\s+is\s+(not\s+)?([A-Za-z][\w-]*)", item.text):
                (negatives if negated else positives).add(f"{subject.lower()}:{value.lower()}")
        return sorted(positives & negatives)
