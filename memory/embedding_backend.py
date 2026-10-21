"""Configurable embedding providers behind PRIMA's stable embedding shim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import tracemalloc
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from config.runtime_config import RuntimeConfig

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
    revision: str | None = None
    device: str = "cpu"
    local_snapshot: str | None = None
    batch_size: int = 16
    cpu_threads: int = 2

    @classmethod
    def from_runtime_config(cls, config: RuntimeConfig) -> EmbeddingBackendConfig:
        name = _resolve_backend_name(config.embedding_backend)
        return cls(name, config.embedding_model or MODEL_BY_BACKEND[name], config.embedding_dimensions,
                   revision=config.embedding_revision, device=config.embedding_device,
                   local_snapshot=config.embedding_local_snapshot, batch_size=config.embedding_batch_size,
                   cpu_threads=config.retrieval_cpu_threads)

    def __post_init__(self) -> None:
        if self.dimensions < 1 or self.batch_size < 1 or self.cpu_threads < 1:
            raise ValueError("Embedding dimensions, batch size and CPU threads must be positive")
        if self.device == "auto" or not self.device:
            raise ValueError("Embedding device must be explicit")


_CONFIG_OVERRIDE: ContextVar[EmbeddingBackendConfig | None] = ContextVar(
    "prima_embedding_backend_config", default=None
)


@contextmanager
def use_embedding_backend_config(config: EmbeddingBackendConfig) -> Iterator[None]:
    """Apply one request-scoped embedding configuration without mutating the process environment."""

    token = _CONFIG_OVERRIDE.set(config)
    try:
        yield
    finally:
        _CONFIG_OVERRIDE.reset(token)


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
            self.model_identifier, revision=config.revision, device=config.device,
            local_snapshot=config.local_snapshot, cpu_threads=config.cpu_threads,
        )
        native = self._model.get_sentence_embedding_dimension()
        if native != self.dimensions:
            raise ValueError(f"Native embedding dimension {native} does not match configured {self.dimensions}; rebuild with the native dimension")
        if config.device == "cpu" and str(self._model.device) != "cpu":
            raise RuntimeError("Required CPU embedding model loaded on another device")

    def embed(self, text: str) -> list[float]:
        vector = np.asarray(self._model.encode(text, normalize_embeddings=True, batch_size=self.config.batch_size,
                                               show_progress_bar=False, convert_to_numpy=True), dtype="float32")
        norm = float(np.linalg.norm(vector))
        if vector.shape != (self.dimensions,) or not np.isfinite(vector).all() or not math.isfinite(norm) or norm == 0:
            raise ValueError("Learned embedding must be a finite, nonzero vector of the native dimension")
        return _normalize(vector.tolist())

    def info(self) -> dict[str, Any]:
        return {**super().info(), "revision": self.config.revision, "device": str(self._model.device),
                "batch_size": self.config.batch_size, "cpu_threads": self.config.cpu_threads,
                "local_files_only": True, "trust_remote_code": False,
                "snapshot": getattr(self._model, "_prima_snapshot_identity", None)}

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
    override = _CONFIG_OVERRIDE.get()
    if override is not None:
        return (
            override
            if dimensions is None or dimensions == override.dimensions
            else replace(override, dimensions=dimensions)
        )
    configured = os.getenv("PRIMA_EMBEDDING_BACKEND", "stable").lower().strip()
    name = _resolve_backend_name(configured)
    configured_model = os.getenv("PRIMA_EMBEDDING_MODEL", "").strip()
    model = configured_model or MODEL_BY_BACKEND[name]
    if name == "stable":
        model = MODEL_BY_BACKEND["stable"]
    return EmbeddingBackendConfig(name=name, model_identifier=model,
        dimensions=dimensions if dimensions is not None else _configured_dimensions(64 if name == "stable" else 384),
        revision=os.getenv("PRIMA_EMBEDDING_REVISION") or None, device=os.getenv("PRIMA_EMBEDDING_DEVICE", "cpu"),
        local_snapshot=os.getenv("PRIMA_EMBEDDING_LOCAL_SNAPSHOT") or None,
        batch_size=int(os.getenv("PRIMA_EMBEDDING_BATCH_SIZE", "16")), cpu_threads=int(os.getenv("PRIMA_RETRIEVAL_CPU_THREADS", "2")))


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
def _sentence_transformer_model(
    model_identifier: str, *, revision: str | None = None, device: str = "cpu",
    local_snapshot: str | None = None, cpu_threads: int = 2,
) -> Any:
    path, identity = model_snapshot(model_identifier, revision, local_snapshot)
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "cpu":
        torch.set_num_threads(cpu_threads)
    model: Any = SentenceTransformer(path, device=device, trust_remote_code=False, local_files_only=True)
    model._prima_snapshot_identity = identity
    return model


def model_snapshot(model_identifier: str, revision: str | None, local_snapshot: str | None = None) -> tuple[str, dict[str, Any]]:
    """Resolve staged files only and fingerprint the exact artifacts handed to a loader."""
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Learned models require an immutable 40-character revision")
    if local_snapshot:
        path = Path(local_snapshot)
        manifest = json.loads((path / "prima_model_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("model_identifier") != model_identifier or manifest.get("revision") != revision:
            raise ValueError("Local model manifest identity does not match the selected model/revision")
    else:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model_identifier, revision=revision, local_files_only=True))
        manifest = None
    if not path.is_dir():
        raise FileNotFoundError(f"Required model snapshot is missing: {path}")
    hashes = {}
    for file in sorted(path.rglob("*")):
        if not file.is_file() or file.name == "prima_model_manifest.json" or any(part.startswith(".") or part == "__pycache__" for part in file.relative_to(path).parts):
            continue
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[file.relative_to(path).as_posix()] = digest.hexdigest()
    if not hashes or not (path / "config.json").is_file():
        raise ValueError("Model snapshot has no loadable configuration/artifacts")
    if manifest is not None and manifest.get("files") != hashes:
        raise ValueError("Local model manifest file hashes do not match the staged snapshot")
    return str(path), {"model_identifier": model_identifier, "revision": revision, "files": hashes,
                       "artifact_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


@lru_cache(maxsize=1)
def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


def _configured_dimensions(default: int = DEFAULT_DIMENSIONS) -> int:
    try:
        dimensions = int(os.getenv("PRIMA_EMBEDDING_DIMENSIONS", str(default)))
    except ValueError as exc:
        raise ValueError("PRIMA_EMBEDDING_DIMENSIONS must be an integer") from exc
    if dimensions <= 0:
        raise ValueError("PRIMA_EMBEDDING_DIMENSIONS must be greater than zero")
    return dimensions


def _resolve_backend_name(configured: str) -> str:
    name = BACKEND_ALIASES.get(configured, configured)
    if name not in MODEL_BY_BACKEND:
        valid_names = ", ".join(sorted(MODEL_BY_BACKEND | BACKEND_ALIASES))
        raise ValueError(f"Unknown PRIMA_EMBEDDING_BACKEND={configured!r}. Valid backends: {valid_names}")
    return name


def _normalize(values: list[float]) -> list[float]:
    vector = np.array(values, dtype="float32")
    norm = np.linalg.norm(vector)
    if norm:
        vector = vector / norm
    return [float(value) for value in vector.tolist()]


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
        if backend.learned:
            validation = validate_learned_backend(backend)
            passed = passed and validation["passed"]
            print(json.dumps(validation, sort_keys=True))

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


def validate_learned_backend(backend: EmbeddingBackend | None = None) -> dict[str, Any]:
    """Opt-in local-model preflight with held-out paraphrases, not benchmark answer hints."""
    backend = backend or get_embedding_backend()
    if not backend.learned:
        raise ValueError("Semantic validation requires an actual learned backend")
    probes = (
        ("The physician treated the patient.", "A doctor cared for someone who was ill.", "A spacecraft orbits a distant planet."),
        ("A puppy is running through grass.", "A young dog races across a field.", "The quarterly financial report was published."),
    )
    rankings = []
    for query, related, distractor in probes:
        vectors = [backend.embed(text) for text in (query, related, distractor)]
        rankings.append({"paraphrase": _cosine_similarity(vectors[0], vectors[1]),
                         "distractor": _cosine_similarity(vectors[0], vectors[2])})
    passed = all(row["paraphrase"] > row["distractor"] for row in rankings)
    if not passed:
        raise RuntimeError(f"Learned paraphrase/ranking sanity check failed: {rankings}")
    return {"passed": passed, "probe_version": "held-out-paraphrases-v1", "rankings": rankings, "backend": backend.info()}


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
