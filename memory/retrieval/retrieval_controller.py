"""Retrieval controller that exposes structured APIs only."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from memory.memory_repository import MemoryRepository
from memory.retrieval.dense_strategy import DenseRetrievalStrategy
from memory.retrieval.hybrid_fusion import HybridFusion
from memory.retrieval.hybrid_fusion import HybridFusionConfig
from memory.retrieval.reranker import Reranker
from memory.retrieval.retrieval_confidence import RetrievalConfidence, RetrievalConfidenceEstimator
from memory.retrieval.retrieval_request import RetrievalRequest
from memory.retrieval.retrieval_result import RetrievalResult
from memory.retrieval.retrieval_strategy import RetrievalStrategy
from memory.retrieval.sparse_strategy import SparseRetrievalStrategy
from memory.retrieval.temporal_strategy import TemporalRetrievalStrategy


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    results: tuple[RetrievalResult, ...]
    confidence: RetrievalConfidence


class RetrievalController:
    def __init__(
        self,
        repository: MemoryRepository,
        strategies: list[RetrievalStrategy] | None = None,
        fusion: HybridFusion | None = None,
        confidence_estimator: RetrievalConfidenceEstimator | None = None,
        reranker: Reranker | None = None,
        candidate_pool_multiplier: int = 12,
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
        self.candidate_pool_multiplier = max(1, candidate_pool_multiplier)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        candidate_request = replace(request, top_k=max(request.top_k, request.top_k * self.candidate_pool_multiplier))
        by_strategy = {
            strategy.name: strategy.retrieve(candidate_request, self.repository)
            for strategy in self.strategies
        }
        fused = self.fusion.fuse(by_strategy, top_k=candidate_request.top_k)
        fused = self._apply_state_filter(fused, request)
        reranked = self.reranker.rerank(fused, request)[: request.top_k]
        confidence = self.confidence_estimator.estimate(reranked, request.top_k)
        return RetrievalResponse(results=tuple(reranked), confidence=confidence)

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
