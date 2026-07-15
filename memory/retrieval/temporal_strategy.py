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
        if not self._should_apply(request):
            return []
        now = datetime.now(timezone.utc)
        horizon = WINDOW_HOURS[request.temporal_window]
        results: list[RetrievalResult] = []
        for memory_type in request.memory_types:
            for note in repository.list(memory_type):
                timestamp = note.timestamp.replace(tzinfo=timezone.utc) if note.timestamp.tzinfo is None else note.timestamp
                age_hours = max(0.0, (now - timestamp).total_seconds() / 3600.0)
                if age_hours > horizon:
                    continue
                recency_score = max(0.0, 1.0 - (age_hours / horizon)) if horizon else 0.0
                constraint_score = self._constraint_score(request, note.content.lower())
                score = max(recency_score * 0.35, constraint_score)
                results.append(
                    RetrievalResult(
                        note=note,
                        score=score,
                        strategy_scores={self.name: score},
                        explanation={
                            "age_hours": round(age_hours, 6),
                            "window": request.temporal_window.value,
                            "constraint_score": round(constraint_score, 6),
                        },
                    )
                )
        return sorted(results, key=lambda item: item.score, reverse=True)[: request.top_k]

    def _should_apply(self, request: RetrievalRequest) -> bool:
        analysis = request.analyzed_query
        if request.temporal_window != RetrievalWindow.LONG_TERM:
            return True
        if analysis is None:
            return True
        return bool(analysis.temporal_expressions or analysis.temporal_constraints or "temporal" in analysis.intents)

    def _constraint_score(self, request: RetrievalRequest, memory_text: str) -> float:
        analysis = request.analyzed_query
        if analysis is None or not analysis.temporal_constraints:
            return 0.0
        matches = 0
        for constraint in analysis.temporal_constraints:
            terms = {constraint.operator, constraint.expression, constraint.anchor or ""}
            if any(term and term in memory_text for term in terms):
                matches += 1
        return matches / max(1, len(analysis.temporal_constraints))
