"""Explicit lexical or cross-encoder reranking backends."""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from enum import Enum
from typing import Any

from memory.retrieval.retrieval_result import RetrievalResult

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")
STOPWORDS = {
    "the", "and", "for", "with", "which", "what", "when", "where", "about",
    "from", "into", "that", "this", "should", "memory", "note", "query",
}


class RerankerBackend(str, Enum):
    """Stable reranker identities reported in manifests and diagnostics."""

    DISABLED = "disabled"
    LEXICAL_FALLBACK = "lexical_fallback"
    CROSS_ENCODER = "cross_encoder"


class Reranker:
    """Rerank with one explicitly selected backend; never silently switch."""

    def __init__(
        self,
        enabled: bool | None = None,
        model_name: str | None = None,
        backend: RerankerBackend | str | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        configured_enabled = _env_bool("PRIMA_RERANKER_ENABLED", True) if enabled is None else enabled
        legacy_cross = _env_bool("PRIMA_CROSS_ENCODER_ENABLED", False)
        configured_backend = backend or os.getenv("PRIMA_RERANKER_BACKEND")
        if not configured_enabled:
            configured_backend = RerankerBackend.DISABLED
        elif configured_backend is None:
            configured_backend = RerankerBackend.CROSS_ENCODER if legacy_cross else RerankerBackend.LEXICAL_FALLBACK
        self.requested_backend = RerankerBackend(configured_backend)
        self.model_name: str = model_name or os.getenv(
            "PRIMA_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ) or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._model_loader = model_loader
        self._cross_encoder: Any | None = None
        self._load_attempted = False

    @property
    def active_backend(self) -> RerankerBackend:
        """Return the backend that will process the next request."""
        return self.requested_backend

    def preflight(self) -> dict[str, Any]:
        """Fail before execution when a required cross-encoder cannot load."""
        if self.requested_backend is RerankerBackend.CROSS_ENCODER:
            self._load_cross_encoder()
        return self.diagnostics()

    def diagnostics(self) -> dict[str, Any]:
        """Return stable requested/active backend identity."""
        return {
            "requested_backend": self.requested_backend.value,
            "active_backend": self.active_backend.value,
            "model": self.model_name if self.active_backend is RerankerBackend.CROSS_ENCODER else None,
            "fallback": self.active_backend is RerankerBackend.LEXICAL_FALLBACK,
        }

    def rerank(self, results: list[RetrievalResult], request: Any | None = None) -> list[RetrievalResult]:
        required = getattr(request, "required_reranker_backend", None)
        if required is not None and RerankerBackend(str(required)) is not self.active_backend:
            raise RuntimeError(
                f"Required reranker backend {required!r} does not match configured "
                f"backend {self.active_backend.value!r}."
            )
        if self.active_backend is RerankerBackend.DISABLED:
            return results
        query = str(getattr(request, "query", "")) if request is not None else ""
        cross_scores = self._cross_encoder_scores(query, results)
        scored: list[RetrievalResult] = []
        for index, result in enumerate(results):
            rerank_score = cross_scores[index] if cross_scores is not None else self._lexical_score(query, result)
            explanation = dict(result.explanation)
            explanation["reranker"] = {
                "backend": self.active_backend.value,
                "model": self.model_name if cross_scores is not None else None,
                "score": round(float(rerank_score), 6),
            }
            strategy_scores = dict(result.strategy_scores)
            strategy_scores["reranker"] = round(float(rerank_score), 6)
            scored.append(replace(result, strategy_scores=strategy_scores, explanation=explanation))
        return sorted(
            scored,
            key=lambda result: (
                result.strategy_scores.get("reranker", 0.0), result.score,
                result.note.salience_score, result.note.retention_score,
            ),
            reverse=True,
        )

    def _cross_encoder_scores(self, query: str, results: list[RetrievalResult]) -> list[float] | None:
        if self.active_backend is not RerankerBackend.CROSS_ENCODER or not query or not results:
            return None
        model = self._load_cross_encoder()
        values = [float(score) for score in model.predict([(query, result.note.content) for result in results])]
        if not values:
            raise RuntimeError(f"Cross-encoder {self.model_name!r} returned no scores.")
        if len(values) != len(results):
            raise RuntimeError(f"Cross-encoder {self.model_name!r} returned an invalid score count.")
        minimum, maximum = min(values), max(values)
        if maximum <= minimum:
            return [0.5 for _ in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

    def _load_cross_encoder(self) -> Any:
        if self._cross_encoder is not None:
            return self._cross_encoder
        if self._load_attempted:
            raise RuntimeError(f"Required cross-encoder {self.model_name!r} is unavailable.")
        self._load_attempted = True
        try:
            if self._model_loader is not None:
                self._cross_encoder = self._model_loader(self.model_name)
            else:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(self.model_name, local_files_only=True)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Required cross-encoder {self.model_name!r} failed preflight: {exc}") from exc
        if self._cross_encoder is None:
            raise RuntimeError(f"Required cross-encoder {self.model_name!r} is unavailable.")
        return self._cross_encoder

    def _lexical_score(self, query: str, result: RetrievalResult) -> float:
        if not query:
            return result.score
        query_terms, note_terms = _tokens(query), _tokens(result.note.content)
        if not query_terms or not note_terms:
            return result.score
        query_counts, note_counts = Counter(query_terms), Counter(note_terms)
        overlap = sum(min(query_counts[token], note_counts[token]) for token in query_counts)
        coverage = overlap / max(1, len(query_counts))
        note_coverage = overlap / max(1, len(note_counts))
        keyword_overlap = len(set(query_terms) & set(result.note.keywords)) / max(1, len(set(query_terms)))
        return result.score * 0.45 + coverage * 0.35 + note_coverage * 0.10 + keyword_overlap * 0.10


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS and len(token) > 2]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}
