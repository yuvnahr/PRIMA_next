"""Classifier and similarity backend interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
from chromadb import logger

from affect.emotion_profile import EmotionProfile


class EmotionClassifier(Protocol):
    def classify(self, text: str) -> EmotionProfile:
        """Classify text and return an immutable profile."""


class SimilarityBackend(ABC):
    """Backend abstraction for lexicon nearest-neighbor search."""

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return scores and indices for the nearest neighbors."""


class NumpyBackend(SimilarityBackend):
    """Deterministic default cosine-similarity backend."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix.astype("float32")

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self.matrix @ query.astype("float32")
        if len(scores) <= top_k:
            indices = np.argsort(scores)[::-1]
        else:
            indices = np.argpartition(scores, -top_k)[-top_k:]
            indices = indices[np.argsort(scores[indices])[::-1]]
        return scores[indices], indices


class FaissBackend(SimilarityBackend):
    """Optional FAISS backend. FAISS is not required for installation."""

    def __init__(self, matrix: np.ndarray) -> None:
        import faiss  # type: ignore[import-not-found]

        self._faiss = faiss
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix.astype("float32"))

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, indices = self.index.search(query.reshape(1, -1).astype("float32"), top_k)
        return scores[0], indices[0]


def choose_similarity_backend(matrix: np.ndarray, prefer_faiss: bool = False) -> SimilarityBackend:
    if prefer_faiss:
        try:
            return FaissBackend(matrix)
        except Exception as e:
            logger.warning(
                "FAISS backend unavailable, falling back to NumPy: %s",
                e
            )
    return NumpyBackend(matrix)
