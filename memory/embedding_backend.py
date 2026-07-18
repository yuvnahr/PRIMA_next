"""Configurable embedding providers behind PRIMA's stable embedding shim."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
import tracemalloc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

DEFAULT_DIMENSIONS = 64
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BENCHMARK_SENTENCE_COUNT = 100
SELF_TEST_SENTENCES = (
    "PRIMA stores durable user memories for later retrieval.",
    "PRIMA stores durable user memories for later retrieval.",
    "A recipe for sourdough bread needs flour and water.",
)


@dataclass(frozen=True, slots=True)
class EmbeddingBackendConfig:
    name: str
    model_identifier: str
    dimensions: int = DEFAULT_DIMENSIONS


class EmbeddingBackend(ABC):
    name: str
    deterministic: bool
    learned: bool
    source_file = "memory/embedding_backend.py"

    def __init__(self, config: EmbeddingBackendConfig) -> None:
        self.config = config

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @property
    def model_identifier(self) -> str:
        return self.config.model_identifier

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    def available(self) -> bool:
        return True

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model_identifier": self.model_identifier,
            "embedding_dimensions": self.dimensions,
            "deterministic": self.deterministic,
            "learned": self.learned,
            "available": self.available(),
            "source_file": self.source_file,
        }


class StableHashEmbeddingBackend(EmbeddingBackend):
    name = "stable"
    deterministic = True
    learned = False

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        while len(values) < self.dimensions:
            for byte in digest:
                values.append((byte / 127.5) - 1.0)
                if len(values) == self.dimensions:
                    break
            digest = hashlib.sha256(digest).digest()
        return _normalize(values)


class SentenceTransformerEmbeddingBackend(EmbeddingBackend):
    name = "sentence_transformers"
    deterministic = False
    learned = True

    def __init__(self, config: EmbeddingBackendConfig) -> None:
        super().__init__(config)
        self.name = config.name
        self._model = _sentence_transformer_model(
            self.model_identifier,
            _requires_trust_remote_code(config.name),
        )

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return _resize(_normalize([float(value) for value in vector]), self.dimensions)

    def available(self) -> bool:
        return _sentence_transformers_available()


BACKEND_ALIASES = {
    "current": "stable",
    "hash": "stable",
    "stable_hash": "stable",
    "stable_hash_embedding": "stable",
    "sentence-transformers": "sentence_transformers",
    "semantic": "sentence_transformers",
    "minilm": "minilm",
    "all-minilm-l6-v2": "minilm",
    "bge-small": "bge_small",
    "bge-small-en-v1.5": "bge_small",
    "nomic": "nomic",
    "nomic-embed-text-v1.5": "nomic",
}

MODEL_BY_BACKEND = {
    "stable": "stable_hash_embedding",
    "sentence_transformers": DEFAULT_MODEL,
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge_small": "BAAI/bge-small-en-v1.5",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
}


def embed_text(text: str, dimensions: int | None = None) -> list[float]:
    """Embed text with the configured provider."""

    return get_embedding_backend(dimensions=dimensions).embed(text)


def stable_hash_embedding(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    """Expose the frozen deterministic baseline for audits and tests."""

    return StableHashEmbeddingBackend(EmbeddingBackendConfig("stable", MODEL_BY_BACKEND["stable"], dimensions)).embed(text)


def get_embedding_backend(dimensions: int | None = None) -> EmbeddingBackend:
    config = embedding_backend_config(dimensions)
    if config.name == "stable":
        return StableHashEmbeddingBackend(config)
    return SentenceTransformerEmbeddingBackend(config)


def embedding_backend_config(dimensions: int | None = None) -> EmbeddingBackendConfig:
    configured = os.getenv("PRIMA_EMBEDDING_BACKEND", "stable").lower().strip()
    name = _resolve_backend_name(configured)
    configured_model = os.getenv("PRIMA_EMBEDDING_MODEL", "").strip()
    model = configured_model or MODEL_BY_BACKEND[name]
    if name == "stable":
        model = MODEL_BY_BACKEND["stable"]
    return EmbeddingBackendConfig(name=name, model_identifier=model, dimensions=dimensions or _configured_dimensions())


def embedding_backend_info() -> dict[str, Any]:
    backend = get_embedding_backend()
    info = backend.info()
    info.update(
        {
            "configured_backend": os.getenv("PRIMA_EMBEDDING_BACKEND", "stable"),
            "configured_model": os.getenv("PRIMA_EMBEDDING_MODEL", ""),
            "semantic_backend_available": _sentence_transformers_available(),
            "fallback_behavior": "none; unavailable learned backends raise during embedding",
        }
    )
    return info


def reset_embedding_backend_cache() -> None:
    _sentence_transformer_model.cache_clear()
    _sentence_transformers_available.cache_clear()


@lru_cache(maxsize=4)
def _sentence_transformer_model(model_identifier: str, trust_remote_code: bool = False) -> Any:
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_identifier, trust_remote_code=trust_remote_code)
    except TypeError:
        if trust_remote_code:
            raise
        return SentenceTransformer(model_identifier)


@lru_cache(maxsize=1)
def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


def _configured_dimensions() -> int:
    try:
        dimensions = int(os.getenv("PRIMA_EMBEDDING_DIMENSIONS", str(DEFAULT_DIMENSIONS)))
    except ValueError:
        raise ValueError("PRIMA_EMBEDDING_DIMENSIONS must be an integer")
    if dimensions <= 0:
        raise ValueError("PRIMA_EMBEDDING_DIMENSIONS must be greater than zero")
    return dimensions


def _resolve_backend_name(configured: str) -> str:
    name = BACKEND_ALIASES.get(configured, configured)
    if name not in MODEL_BY_BACKEND:
        valid_names = ", ".join(sorted(MODEL_BY_BACKEND | BACKEND_ALIASES))
        raise ValueError(f"Unknown PRIMA_EMBEDDING_BACKEND={configured!r}. Valid backends: {valid_names}")
    return name


def _requires_trust_remote_code(backend_name: str) -> bool:
    return backend_name == "nomic"


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


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=False)) / (left_norm * right_norm)


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def _run_self_test() -> int:
    try:
        backend = get_embedding_backend()
        started = time.perf_counter()
        embeddings = [backend.embed(sentence) for sentence in SELF_TEST_SENTENCES]
        elapsed = time.perf_counter() - started
        identical_similarity = _cosine_similarity(embeddings[0], embeddings[1])
        different_topic_similarity = _cosine_similarity(embeddings[0], embeddings[2])
        sanity_passed = identical_similarity > different_topic_similarity
        dimensions = len(embeddings[0])
        model_loaded = backend.name == "stable" or getattr(backend, "_model", None) is not None
        passed = bool(model_loaded and sanity_passed and all(len(vector) == dimensions for vector in embeddings))

        print(f"active backend: {backend.name}")
        print(f"embedding dimension: {dimensions}")
        print(f"model loaded: {_format_bool(model_loaded)}")
        print(f"average embedding time: {(elapsed / len(SELF_TEST_SENTENCES)) * 1000:.2f} ms")
        print(
            "cosine similarity sanity check: "
            f"identical={identical_similarity:.4f}, different={different_topic_similarity:.4f}"
        )
        print(f"pass/fail: {'pass' if passed else 'fail'}")
        return 0 if passed else 1
    except Exception as exc:
        print(f"active backend: {os.getenv('PRIMA_EMBEDDING_BACKEND', 'stable')}")
        print("embedding dimension: unavailable")
        print("model loaded: no")
        print("average embedding time: unavailable")
        print(f"cosine similarity sanity check: failed ({exc})")
        print("pass/fail: fail")
        return 1


def _benchmark_sentences() -> list[str]:
    return [
        f"Benchmark sentence {index}: PRIMA validates learned embedding backend latency and retrieval vectors."
        for index in range(BENCHMARK_SENTENCE_COUNT)
    ]


def _run_benchmark() -> int:
    try:
        backend = get_embedding_backend()
        sentences = _benchmark_sentences()
        tracemalloc.start()
        started = time.perf_counter()
        embeddings = [backend.embed(sentence) for sentence in sentences]
        elapsed = time.perf_counter() - started
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if len(embeddings) != len(sentences):
            raise RuntimeError("benchmark did not produce the expected number of embeddings")
        average_latency = elapsed / len(sentences)
        throughput = len(sentences) / elapsed if elapsed else float("inf")

        print(f"active backend: {backend.name}")
        print(f"embedding dimension: {len(embeddings[0]) if embeddings else 0}")
        print(f"average latency: {average_latency * 1000:.2f} ms")
        print(f"throughput: {throughput:.2f} embeddings/sec")
        print(f"memory usage: current={current_bytes / 1024 / 1024:.2f} MiB, peak={peak_bytes / 1024 / 1024:.2f} MiB")
        return 0
    except Exception as exc:
        print(f"benchmark failed: {exc}")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate PRIMA embedding backends.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true", help="Run a lightweight backend validation.")
    mode.add_argument("--benchmark", action="store_true", help="Embed 100 sentences and report backend performance.")
    args = parser.parse_args(argv)
    if args.self_test:
        return _run_self_test()
    return _run_benchmark()


if __name__ == "__main__":
    sys.exit(main())
