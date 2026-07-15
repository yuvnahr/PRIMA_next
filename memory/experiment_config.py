"""Configuration for controlled scientific ablations."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from memory.embedding_backend import reset_embedding_backend_cache


@dataclass(frozen=True, slots=True)
class EmbeddingExperimentConfig:
    backend: str = "stable"
    representation_mode: str = "semantic"
    identity_enabled: bool = True

    def __post_init__(self) -> None:
        if self.representation_mode not in {"raw", "semantic", "event"}:
            raise ValueError("representation_mode must be raw, semantic, or event")


@contextmanager
def configured_backend(name: str) -> Iterator[None]:
    old = os.environ.get("PRIMA_EMBEDDING_BACKEND")
    os.environ["PRIMA_EMBEDDING_BACKEND"] = name
    reset_embedding_backend_cache()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("PRIMA_EMBEDDING_BACKEND", None)
        else:
            os.environ["PRIMA_EMBEDDING_BACKEND"] = old
        reset_embedding_backend_cache()
