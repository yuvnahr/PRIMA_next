"""Optional cross-encoder reranker with deterministic local fallback."""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import replace
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
    def __init__(self, enabled: bool | None = None, model_name: str | None = None) -> None:
        self.enabled = _env_bool("PRIMA_RERANKER_ENABLED", True) if enabled is None else enabled
        self.model_name = model_name or os.getenv("PRIMA_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.cross_encoder_enabled = _env_bool("PRIMA_CROSS_ENCODER_ENABLED", False)
        self._cross_encoder: Any | None = None
        self._load_attempted = False

    def rerank(self, results: list[RetrievalResult], request: Any | None = None) -> list[RetrievalResult]:
        if not self.enabled:
            return results
        query = str(getattr(request, "query", "")) if request is not None else ""
        cross_scores = self._cross_encoder_scores(query, results)
        scored: list[RetrievalResult] = []
        for index, result in enumerate(results):
            fallback_score = self._semantic_score(query, result)
            rerank_score = cross_scores[index] if cross_scores is not None else fallback_score
            combined = max(0.0, min(1.0, result.score * 0.45 + rerank_score * 0.55))
            explanation = dict(result.explanation)
            explanation["reranker"] = {
                "enabled": True,
                "model": self.model_name if cross_scores is not None else "lexical_fallback",
                "score": round(float(rerank_score), 6),
            }
            strategy_scores = dict(result.strategy_scores)
            strategy_scores["reranker"] = round(float(rerank_score), 6)
            scored.append(replace(result, score=combined, strategy_scores=strategy_scores, explanation=explanation))
        return sorted(
            scored,
            key=lambda result: (result.score, result.note.salience_score, result.note.retention_score),
            reverse=True,
        )

    def _cross_encoder_scores(self, query: str, results: list[RetrievalResult]) -> list[float] | None:
        if not self.cross_encoder_enabled or not query or not results:
            return None
        model = self._load_cross_encoder()
        if model is None:
            return None
        raw_scores = model.predict([(query, result.note.content) for result in results])
        values = [float(score) for score in raw_scores]
        if not values:
            return None
        minimum = min(values)
        maximum = max(values)
        if maximum <= minimum:
            return [0.5 for _ in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

    def _load_cross_encoder(self) -> Any | None:
        if self._load_attempted:
            return self._cross_encoder
        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.model_name)
        except Exception:
            self._cross_encoder = None
        return self._cross_encoder

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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


