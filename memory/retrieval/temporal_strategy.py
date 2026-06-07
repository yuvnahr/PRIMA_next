"""Temporal retrieval strategy."""

from __future__ import annotations

from datetime import datetime, timezone

from memory.memory_repository import MemoryRepository
from memory.memory_types import RetrievalWindow
from memory.retrieval.retrieval_request import RetrievalRequest
from memory.retrieval.retrieval_result import RetrievalResult
from memory.retrieval.retrieval_strategy import RetrievalStrategy


WINDOW_HOURS = {
    RetrievalWindow.LAST_HOUR: 1,
    RetrievalWindow.LAST_DAY: 24,
    RetrievalWindow.LAST_WEEK: 24 * 7,
    RetrievalWindow.LONG_TERM: 24 * 365 * 10,
}


class TemporalRetrievalStrategy(RetrievalStrategy):
    name = "temporal"

    def retrieve(self, request: RetrievalRequest, repository: MemoryRepository) -> list[RetrievalResult]:
        now = datetime.now(timezone.utc)
        horizon = WINDOW_HOURS[request.temporal_window]
        results: list[RetrievalResult] = []
        for memory_type in request.memory_types:
            for note in repository.list(memory_type):
                age_hours = max(0.0, (now - note.timestamp).total_seconds() / 3600.0)
                if age_hours > horizon:
                    continue
                score = max(0.0, 1.0 - (age_hours / horizon)) if horizon else 0.0
                results.append(
                    RetrievalResult(
                        note=note,
                        score=score,
                        strategy_scores={self.name: score},
                        explanation={"age_hours": round(age_hours, 6), "window": request.temporal_window.value},
                    )
                )
        return sorted(results, key=lambda item: item.score, reverse=True)[: request.top_k]
