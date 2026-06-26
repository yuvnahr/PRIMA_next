"""Lightweight deterministic reranker hook."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from memory.retrieval.retrieval_result import RetrievalResult

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "which",
    "what",
    "when",
    "where",
    "about",
    "from",
    "into",
    "that",
    "this",
    "should",
    "memory",
    "note",
    "query",
}


class Reranker:
    def rerank(self, results: list[RetrievalResult], request: Any | None = None) -> list[RetrievalResult]:
        query = str(getattr(request, "query", "")) if request is not None else ""
        return sorted(
            results,
            key=lambda result: (
                self._semantic_score(query, result),
                result.score,
                result.note.salience_score,
                result.note.retention_score,
            ),
            reverse=True,
        )

    def _semantic_score(self, query: str, result: RetrievalResult) -> float:
        if not query:
            return result.score
        query_terms = _tokens(query)
        note_terms = _tokens(result.note.content)
        if not query_terms or not note_terms:
            return result.score
        query_counts = Counter(query_terms)
        note_counts = Counter(note_terms)
        overlap = sum(min(query_counts[token], note_counts[token]) for token in query_counts)
        coverage = overlap / max(1, len(query_counts))
        note_coverage = overlap / max(1, len(note_counts))
        keyword_overlap = len(set(query_terms) & set(result.note.keywords)) / max(1, len(set(query_terms)))
        return result.score * 0.45 + coverage * 0.35 + note_coverage * 0.10 + keyword_overlap * 0.10


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS and len(token) > 2]
