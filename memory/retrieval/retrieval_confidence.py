"""Retrieval confidence estimation from multiple evidence signals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from memory.retrieval.query_analysis import QueryAnalysis, temporal_agreement
from memory.retrieval.retrieval_result import RetrievalResult

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")


@dataclass(frozen=True, slots=True)
class RetrievalConfidence:
    confidence: float
    ambiguity_score: float
    coverage_score: float
    retrieval_quality_score: float
    lexical_overlap_score: float = 0.0
    entity_overlap_score: float = 0.0
    temporal_agreement_score: float = 0.0
    score_separation: float = 0.0
    reranker_score: float = 0.0
    dense_similarity_score: float = 0.0
    sparse_similarity_score: float = 0.0
    relation_overlap_score: float = 0.0
    candidate_agreement_score: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "confidence": self.confidence,
            "ambiguity_score": self.ambiguity_score,
            "coverage_score": self.coverage_score,
            "retrieval_quality_score": self.retrieval_quality_score,
            "lexical_overlap_score": self.lexical_overlap_score,
            "entity_overlap_score": self.entity_overlap_score,
            "temporal_agreement_score": self.temporal_agreement_score,
            "score_separation": self.score_separation,
            "reranker_score": self.reranker_score,
            "dense_similarity_score": self.dense_similarity_score,
            "sparse_similarity_score": self.sparse_similarity_score,
            "relation_overlap_score": self.relation_overlap_score,
            "candidate_agreement_score": self.candidate_agreement_score,
        }


class RetrievalConfidenceEstimator:
    def estimate(self, results: list[RetrievalResult], requested_k: int, request: Any | None = None) -> RetrievalConfidence:
        if not results:
            return RetrievalConfidence(0.0, 1.0, 0.0, 0.0)

        analysis = getattr(request, "analyzed_query", None) if request is not None else None
        query = str(getattr(request, "query", "")) if request is not None else ""
        lexical_query = str(request.lexical_query()) if request is not None and hasattr(request, "lexical_query") else query
        top_score = results[0].score
        second_score = results[1].score if len(results) > 1 else 0.0
        separation = max(0.0, min(1.0, top_score - second_score))
        ambiguity = max(0.0, min(1.0, 1.0 - separation))
        coverage = max(0.0, min(1.0, len(results) / max(1, requested_k)))
        quality = max(0.0, min(1.0, sum(result.score for result in results) / len(results)))
        lexical = _average(_lexical_overlap(lexical_query, result.note.content) for result in results)
        entity = _average(_entity_overlap(analysis, result.note.content) for result in results)
        relation = _average(_relation_overlap(analysis, result.note.content) for result in results)
        temporal = _average(temporal_agreement(analysis, result.note.timestamp) for result in results) if isinstance(analysis, QueryAnalysis) else 0.5
        reranker = _average(float(result.strategy_scores.get("reranker", 0.0)) for result in results)
        dense = _average(float(result.strategy_scores.get("dense", 0.0)) for result in results if "dense" in result.strategy_scores)
        sparse = _average(float(result.strategy_scores.get("sparse", 0.0)) for result in results if "sparse" in result.strategy_scores)
        candidate_agreement = _average(_candidate_agreement(result) for result in results)
        confidence = (
            quality * 0.18
            + separation * 0.12
            + coverage * 0.08
            + dense * 0.10
            + sparse * 0.12
            + reranker * 0.12
            + entity * 0.10
            + relation * 0.08
            + temporal * 0.06
            + candidate_agreement * 0.04
        )
        return RetrievalConfidence(
            confidence=round(max(0.0, min(1.0, confidence)), 6),
            ambiguity_score=round(ambiguity, 6),
            coverage_score=round(coverage, 6),
            retrieval_quality_score=round(quality, 6),
            lexical_overlap_score=round(lexical, 6),
            entity_overlap_score=round(entity, 6),
            temporal_agreement_score=round(temporal, 6),
            score_separation=round(separation, 6),
            reranker_score=round(reranker, 6),
            dense_similarity_score=round(dense, 6),
            sparse_similarity_score=round(sparse, 6),
            relation_overlap_score=round(relation, 6),
            candidate_agreement_score=round(candidate_agreement, 6),
        )


def _lexical_overlap(query: str, text: str) -> float:
    query_terms = set(_tokens(query))
    text_terms = set(_tokens(text))
    if not query_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def _entity_overlap(analysis: QueryAnalysis | None, text: str) -> float:
    if not isinstance(analysis, QueryAnalysis) or not analysis.entities:
        return 0.5
    lower_text = text.lower()
    matches = 0
    for entity in analysis.entities:
        aliases = analysis.aliases.get(entity, ())
        if any(name.lower() in lower_text for name in (entity, *aliases)):
            matches += 1
    return matches / max(1, len(analysis.entities))


def _relation_overlap(analysis: QueryAnalysis | None, text: str) -> float:
    if not isinstance(analysis, QueryAnalysis):
        return 0.5
    terms = set(analysis.relations) | {term for intent in analysis.intents for term in intent.split()}
    if not terms:
        return 0.5
    text_tokens = set(_tokens(text))
    lower_text = text.lower()
    matches = sum(1 for term in terms if term in text_tokens or term in lower_text)
    return matches / max(1, len(terms))


def _candidate_agreement(result: RetrievalResult) -> float:
    evidence_sources = {"dense", "sparse", "temporal", "semantic", "reranker"}
    active = sum(1 for name in evidence_sources if float(result.strategy_scores.get(name, 0.0) or 0.0) > 0.0)
    return active / len(evidence_sources)


def _tokens(text: str) -> list[str]:
    return [token.lower().strip("'") for token in TOKEN_RE.findall(text) if len(token) > 2]


def _average(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0
