"""Configurable embedding backend behind PRIMA's stable embedding interface."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import Any

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed_text(text: str, dimensions: int = 64) -> list[float]:
    backend = os.getenv("PRIMA_EMBEDDING_BACKEND", "stable").lower()
    if backend in {"sentence_transformers", "sentence-transformers", "semantic"}:
        embedding = _sentence_transformer_embedding(text)
        if embedding is not None:
            return _resize(_normalize(embedding), dimensions)
    return stable_hash_embedding(text, dimensions)


def embedding_backend_info() -> dict[str, Any]:
    backend = os.getenv("PRIMA_EMBEDDING_BACKEND", "stable").lower()
    return {
        "backend": backend,
        "model": os.getenv("PRIMA_EMBEDDING_MODEL", DEFAULT_MODEL) if backend != "stable" else "stable_hash_embedding",
        "semantic_backend_available": _sentence_transformers_available() if backend != "stable" else False,
    }


def stable_hash_embedding(text: str, dimensions: int = 64) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < dimensions:
        for byte in digest:
            values.append((byte / 127.5) - 1.0)
            if len(values) == dimensions:
                break
        digest = hashlib.sha256(digest).digest()
    return _normalize(values)


def _sentence_transformer_embedding(text: str) -> list[float] | None:
    if not _sentence_transformers_available():
        return None
    try:
        model = _sentence_transformer_model()
        vector = model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]
    except Exception:
        return None


@lru_cache(maxsize=1)
def _sentence_transformer_model() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(os.getenv("PRIMA_EMBEDDING_MODEL", DEFAULT_MODEL))


@lru_cache(maxsize=1)
def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


def _normalize(values: list[float]) -> list[float]:
    vector = np.array(values, dtype="float32")
    norm = np.linalg.norm(vector)
    if norm:
        vector = vector / norm
    return [float(value) for value in vector.tolist()]


def _resize(values: list[float], dimensions: int) -> list[float]:
    if len(values) == dimensions:
        return values
    if len(values) > dimensions:
        return _normalize(values[:dimensions])
    padded = list(values)
    padded.extend(0.0 for _ in range(dimensions - len(padded)))
    return _normalize(padded)
