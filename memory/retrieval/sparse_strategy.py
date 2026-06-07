"""Sparse dominant context chain retrieval strategy."""

from __future__ import annotations

from memory.memory_note import calculate_smart_overlap, extract_dominant_context_chain
from memory.memory_repository import MemoryRepository
from memory.retrieval.retrieval_request import RetrievalRequest
from memory.retrieval.retrieval_result import RetrievalResult
from memory.retrieval.retrieval_strategy import RetrievalStrategy


class SparseRetrievalStrategy(RetrievalStrategy):
    name = "sparse"

    def retrieve(self, request: RetrievalRequest, repository: MemoryRepository) -> list[RetrievalResult]:
        query_chain = extract_dominant_context_chain(request.query)
        results: list[RetrievalResult] = []
        for memory_type in request.memory_types:
            for note in repository.list(memory_type):
                keywords = note.keywords
                overlap = calculate_smart_overlap(query_chain, keywords)
                if overlap <= 0:
                    continue
                score = min(1.0, overlap / max(1, len(query_chain)))
                results.append(
                    RetrievalResult(
                        note=note,
                        score=score,
                        strategy_scores={self.name: score},
                        explanation={"query_chain": query_chain, "overlap": overlap},
                    )
                )
        return sorted(results, key=lambda item: item.score, reverse=True)[: request.top_k]
