"""Lightweight deterministic reranker hook."""

from __future__ import annotations

from memory.retrieval.retrieval_result import RetrievalResult


class Reranker:
    def rerank(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return sorted(
            results,
            key=lambda result: (result.score, result.note.salience_score, result.note.retention_score),
            reverse=True,
        )
