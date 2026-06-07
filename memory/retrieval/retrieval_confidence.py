"""Retrieval confidence estimation."""

from __future__ import annotations

from dataclasses import dataclass

from memory.retrieval.retrieval_result import RetrievalResult


@dataclass(frozen=True, slots=True)
class RetrievalConfidence:
    confidence: float
    ambiguity_score: float
    coverage_score: float
    retrieval_quality_score: float


class RetrievalConfidenceEstimator:
    def estimate(self, results: list[RetrievalResult], requested_k: int) -> RetrievalConfidence:
        if not results:
            return RetrievalConfidence(0.0, 1.0, 0.0, 0.0)

        top_score = results[0].score
        second_score = results[1].score if len(results) > 1 else 0.0
        ambiguity = max(0.0, min(1.0, 1.0 - abs(top_score - second_score)))
        coverage = max(0.0, min(1.0, len(results) / max(1, requested_k)))
        quality = max(0.0, min(1.0, sum(result.score for result in results) / len(results)))
        confidence = max(0.0, min(1.0, quality * 0.55 + coverage * 0.25 + (1.0 - ambiguity) * 0.20))
        return RetrievalConfidence(
            confidence=round(confidence, 6),
            ambiguity_score=round(ambiguity, 6),
            coverage_score=round(coverage, 6),
            retrieval_quality_score=round(quality, 6),
        )
