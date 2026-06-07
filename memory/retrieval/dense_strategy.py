"""Dense vector retrieval strategy."""

from __future__ import annotations

from memory.memory_repository import MemoryRepository
from memory.retrieval.retrieval_request import RetrievalRequest
from memory.retrieval.retrieval_result import RetrievalResult
from memory.retrieval.retrieval_strategy import RetrievalStrategy


class DenseRetrievalStrategy(RetrievalStrategy):
    name = "dense"

    def retrieve(self, request: RetrievalRequest, repository: MemoryRepository) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for memory_type in request.memory_types:
            for note, score in repository.query(request.embedding(), memory_type=memory_type, limit=request.top_k):
                bounded_score = max(0.0, min(1.0, (score + 1.0) / 2.0 if score < 0 else score))
                results.append(
                    RetrievalResult(
                        note=note,
                        score=bounded_score,
                        strategy_scores={self.name: bounded_score},
                    )
                )
        return sorted(results, key=lambda item: item.score, reverse=True)[: request.top_k]
