"""Retrieval controller that exposes structured APIs only."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from memory.memory_repository import MemoryRepository
from memory.retrieval.dense_strategy import DenseRetrievalStrategy
from memory.retrieval.hybrid_fusion import HybridFusion, HybridFusionConfig
from memory.retrieval.query_analysis import QueryAnalyzer, temporal_agreement
from memory.retrieval.reranker import Reranker
from memory.retrieval.retrieval_confidence import RetrievalConfidence, RetrievalConfidenceEstimator
from memory.retrieval.retrieval_request import RetrievalRequest
from memory.retrieval.retrieval_result import RetrievalResult
from memory.retrieval.retrieval_strategy import RetrievalStrategy
from memory.retrieval.sparse_strategy import SparseRetrievalStrategy
from memory.retrieval.temporal_strategy import TemporalRetrievalStrategy

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")
SEMANTIC_BOOST_WEIGHT = 0.08


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    results: tuple[RetrievalResult, ...]
    confidence: RetrievalConfidence
    diagnostics: dict[str, Any] = field(default_factory=dict)


class RetrievalController:
    def __init__(
        self,
        repository: MemoryRepository,
        strategies: list[RetrievalStrategy] | None = None,
        fusion: HybridFusion | None = None,
        confidence_estimator: RetrievalConfidenceEstimator | None = None,
        reranker: Reranker | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        candidate_pool_multiplier: int = 6,
        candidate_pool_size: int = 30,
    ) -> None:
        self.repository = repository
        self.strategies = strategies or [
            DenseRetrievalStrategy(),
            SparseRetrievalStrategy(),
            TemporalRetrievalStrategy(),
        ]
        self.fusion = fusion or HybridFusion(HybridFusionConfig.from_file())
        self.confidence_estimator = confidence_estimator or RetrievalConfidenceEstimator()
        self.reranker = reranker or Reranker()
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.candidate_pool_multiplier = max(1, candidate_pool_multiplier)
        self.candidate_pool_size = max(1, candidate_pool_size)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        request = self._prepare_request(request)
        candidate_top_k = max(request.top_k, self.candidate_pool_size, request.top_k * self.candidate_pool_multiplier)
        candidate_request = replace(request, top_k=candidate_top_k)
        by_strategy = {
            strategy.name: strategy.retrieve(candidate_request, self.repository)
            for strategy in self.strategies
        }
        fused = self.fusion.fuse(by_strategy, top_k=candidate_request.top_k)
        fused = self._apply_state_filter(fused, request)
        fused = self._apply_semantic_boost(fused, request)
        reranked_pool = self.reranker.rerank(fused, request)
        reranked = reranked_pool[: request.top_k]
        confidence = self.confidence_estimator.estimate(reranked, request.top_k, request)
        diagnostics = self._diagnostics(
            request=request,
            by_strategy=by_strategy,
            fused=fused,
            reranked_pool=reranked_pool,
            final_results=reranked,
            confidence=confidence,
        )
        return RetrievalResponse(results=tuple(reranked), confidence=confidence, diagnostics=diagnostics)

    def _prepare_request(self, request: RetrievalRequest) -> RetrievalRequest:
        analysis = request.analyzed_query or self.query_analyzer.analyze(request.query, request.affective_context)
        expanded = request.expanded_query or self.query_analyzer.expand(request.query, analysis)
        return replace(request, analyzed_query=analysis, expanded_query=expanded)

    def _apply_state_filter(self, results: list[RetrievalResult], request: RetrievalRequest) -> list[RetrievalResult]:
        if not request.state_filter:
            return results
        boosted: list[RetrievalResult] = []
        for result in results:
            snapshot = result.note.state_snapshot.to_dict()
            matches = 0
            checks = 0
            for section_name, expected_values in request.state_filter.items():
                if not isinstance(expected_values, dict):
                    continue
                section = snapshot.get(section_name, {})
                if not isinstance(section, dict):
                    continue
                for key, expected in expected_values.items():
                    checks += 1
                    if section.get(key) == expected:
                        matches += 1
            state_score = matches / checks if checks else 0.0
            if state_score:
                strategy_scores = dict(result.strategy_scores)
                strategy_scores["state"] = state_score
                boosted.append(result.with_score(min(1.0, result.score + state_score * 0.1), strategy_scores))
            else:
                boosted.append(result)
        return boosted

    def _apply_semantic_boost(self, results: list[RetrievalResult], request: RetrievalRequest) -> list[RetrievalResult]:
        analysis = request.analyzed_query
        if analysis is None:
            return results
        boosted: list[RetrievalResult] = []
        relation_terms = set(analysis.relations) | set(analysis.intents)
        intent_terms = set(analysis.preference_terms) | set(analysis.identity_attributes) | set(analysis.events)
        for result in results:
            text = result.note.content.lower()
            text_tokens = set(_tokens(text))
            entity_score = _entity_match(analysis.entities, analysis.aliases, text)
            relation_score = _term_match(relation_terms, text_tokens, text)
            intent_score = _term_match(intent_terms, text_tokens, text)
            temporal_score = temporal_agreement(analysis, result.note.timestamp)
            semantic_score = entity_score * 0.30 + relation_score * 0.25 + intent_score * 0.25 + temporal_score * 0.20
            strategy_scores = dict(result.strategy_scores)
            strategy_scores["semantic"] = round(semantic_score, 6)
            if analysis.temporal_constraints:
                strategy_scores["temporal_constraint"] = round(temporal_score, 6)
            explanation = dict(result.explanation)
            explanation["semantic_boost"] = {
                "entity": round(entity_score, 6),
                "relation": round(relation_score, 6),
                "intent": round(intent_score, 6),
                "temporal": round(temporal_score, 6),
                "weight": SEMANTIC_BOOST_WEIGHT,
            }
            boosted.append(replace(result, score=min(1.0, result.score + semantic_score * SEMANTIC_BOOST_WEIGHT), strategy_scores=strategy_scores, explanation=explanation))
        return sorted(boosted, key=lambda item: item.score, reverse=True)

    def _diagnostics(
        self,
        request: RetrievalRequest,
        by_strategy: dict[str, list[RetrievalResult]],
        fused: list[RetrievalResult],
        reranked_pool: list[RetrievalResult],
        final_results: list[RetrievalResult],
        confidence: RetrievalConfidence,
    ) -> dict[str, Any]:
        analysis = request.analyzed_query
        expanded = request.expanded_query
        return {
            "question": request.query,
            "expanded_query": expanded.to_dict() if expanded is not None else {"text": request.query, "terms": [], "expansion_map": {}},
            "query_analysis": analysis.to_dict() if analysis is not None else {},
            "entities": list(analysis.entities) if analysis is not None else [],
            "relations": list(analysis.relations) if analysis is not None else [],
            "intents": list(analysis.intents) if analysis is not None else [],
            "events": list(analysis.events) if analysis is not None else [],
            "temporal_constraints": [constraint.to_dict() for constraint in analysis.temporal_constraints] if analysis is not None else [],
            "temporal_expressions": list(analysis.temporal_expressions) if analysis is not None else [],
            "preference_terms": list(analysis.preference_terms) if analysis is not None else [],
            "identity_attributes": list(analysis.identity_attributes) if analysis is not None else [],
            "expanded_query_terms": list(expanded.terms) if expanded is not None else [],
            "expansion_term_count": len(expanded.terms) if expanded is not None else 0,
            "dense_top30": self._serialize_results(by_strategy.get("dense", ())[:30]),
            "sparse_top30": self._serialize_results(by_strategy.get("sparse", ())[:30]),
            "fused_top30": self._serialize_results(fused[:30]),
            "reranked_top30": self._serialize_results(reranked_pool[:30]),
            "dense_candidates": self._serialize_results(by_strategy.get("dense", ())),
            "sparse_candidates": self._serialize_results(by_strategy.get("sparse", ())),
            "hybrid_candidates": self._serialize_results(fused),
            "reranked_candidates": self._serialize_results(reranked_pool),
            "final_candidates": self._serialize_results(final_results),
            "retrieval_confidence": confidence.to_dict(),
        }

    def _serialize_results(self, results: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": result.note.id,
                "score": round(float(result.score), 6),
                "strategy_scores": {key: round(float(value), 6) for key, value in result.strategy_scores.items()},
                "timestamp": result.note.timestamp.isoformat(),
                "text": result.note.content,
                "explanation": dict(result.explanation),
            }
            for result in results
        ]


def _tokens(text: str) -> list[str]:
    return [token.lower().strip("'") for token in TOKEN_RE.findall(text) if len(token) > 2]


def _entity_match(entities: tuple[str, ...], aliases: dict[str, tuple[str, ...]], text: str) -> float:
    if not entities:
        return 0.5
    matches = 0
    for entity in entities:
        names = (entity, *aliases.get(entity, ()))
        if any(name.lower() in text for name in names):
            matches += 1
    return matches / max(1, len(entities))


def _term_match(terms: set[str], text_tokens: set[str], text: str) -> float:
    clean_terms = {term for term in terms if term}
    if not clean_terms:
        return 0.5
    matches = sum(1 for term in clean_terms if term in text_tokens or term in text)
    return matches / max(1, len(clean_terms))
