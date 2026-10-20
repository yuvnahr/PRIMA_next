"""Typed effective runtime settings resolved once at the public boundary."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _bool(values: Mapping[str, str], name: str, default: bool) -> bool:
    value = values.get(name)
    return default if value is None else value.casefold() in {"1", "true", "yes", "on"}


class RuntimeConfig(BaseModel):
    """Immutable effective affect, embedding, retrieval, and reranker configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    affect_backend: str = "legacy"
    affect_model_id: str | None = None
    affect_model_revision: str | None = None
    affect_device: str = "auto"
    affect_batch_size: int = Field(default=4, gt=0)
    affect_thresholds_path: str | None = None
    affect_calibration_path: str | None = None
    affect_local_files_only: bool = False
    affect_allow_fallback: bool = False
    embedding_backend: str = "stable"
    embedding_model: str | None = None
    embedding_dimensions: int = Field(default=64, gt=0)
    embedding_revision: str | None = None
    embedding_device: str = "cpu"
    embedding_local_snapshot: str | None = None
    embedding_batch_size: int = Field(default=16, gt=0)
    retrieval_cpu_threads: int = Field(default=2, gt=0)
    representation_mode: str = "semantic"
    identity_normalization: bool = True
    retrieval_candidate_pool_size: int = Field(default=30, gt=0)
    retrieval_strategies: tuple[str, ...] | None = None
    retrieval_weights: dict[str, float] | None = None
    reranker_enabled: bool = True
    reranker_backend: str = "lexical_fallback"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_revision: str | None = None
    reranker_device: str = "cpu"
    reranker_local_snapshot: str | None = None
    reranker_batch_size: int = Field(default=8, gt=0)

    @field_validator("retrieval_strategies")
    @classmethod
    def validate_strategies(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is not None and (not value or len(set(value)) != len(value) or set(value) - {"dense", "sparse", "temporal", "graph"}):
            raise ValueError("retrieval_strategies must contain unique supported strategy names")
        return value

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> RuntimeConfig:
        values = os.environ if environment is None else environment
        legacy_cross = _bool(values, "PRIMA_CROSS_ENCODER_ENABLED", False)
        reranker_enabled = _bool(values, "PRIMA_RERANKER_ENABLED", True)
        backend = values.get("PRIMA_RERANKER_BACKEND") or (
            "cross_encoder" if legacy_cross else "lexical_fallback"
        )
        if not reranker_enabled:
            backend = "disabled"
        return cls(
            affect_backend=values.get("PRIMA_AFFECT_BACKEND", "legacy").strip().lower(),
            affect_model_id=values.get("PRIMA_AFFECT_MODEL_ID") or None,
            affect_model_revision=values.get("PRIMA_AFFECT_MODEL_REVISION") or None,
            affect_device=values.get("PRIMA_AFFECT_DEVICE", "auto"),
            affect_batch_size=int(values.get("PRIMA_AFFECT_BATCH_SIZE", "4")),
            affect_thresholds_path=values.get("PRIMA_AFFECT_THRESHOLDS_PATH") or None,
            affect_calibration_path=values.get("PRIMA_AFFECT_CALIBRATION_PATH") or None,
            affect_local_files_only=_bool(values, "PRIMA_AFFECT_LOCAL_FILES_ONLY", False),
            affect_allow_fallback=_bool(values, "PRIMA_AFFECT_ALLOW_FALLBACK", False),
            embedding_backend=values.get("PRIMA_EMBEDDING_BACKEND", "stable").strip().lower(),
            embedding_model=values.get("PRIMA_EMBEDDING_MODEL") or None,
            embedding_dimensions=int(values.get("PRIMA_EMBEDDING_DIMENSIONS", "64" if values.get("PRIMA_EMBEDDING_BACKEND", "stable") in {"stable", "current", "hash", "stable_hash", "stable_hash_embedding"} else "384")),
            embedding_revision=values.get("PRIMA_EMBEDDING_REVISION") or None,
            embedding_device=values.get("PRIMA_EMBEDDING_DEVICE", "cpu"),
            embedding_local_snapshot=values.get("PRIMA_EMBEDDING_LOCAL_SNAPSHOT") or None,
            embedding_batch_size=int(values.get("PRIMA_EMBEDDING_BATCH_SIZE", "16")),
            retrieval_cpu_threads=int(values.get("PRIMA_RETRIEVAL_CPU_THREADS", "2")),
            representation_mode=values.get("PRIMA_REPRESENTATION_MODE", "semantic"),
            identity_normalization=_bool(values, "PRIMA_IDENTITY_NORMALIZATION", True),
            retrieval_candidate_pool_size=int(values.get("PRIMA_RETRIEVAL_CANDIDATE_POOL_SIZE", "30")),
            reranker_enabled=reranker_enabled,
            reranker_backend=backend,
            reranker_model=values.get(
                "PRIMA_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            reranker_revision=values.get("PRIMA_RERANKER_REVISION") or None,
            reranker_device=values.get("PRIMA_RERANKER_DEVICE", "cpu"),
            reranker_local_snapshot=values.get("PRIMA_RERANKER_LOCAL_SNAPSHOT") or None,
            reranker_batch_size=int(values.get("PRIMA_RERANKER_BATCH_SIZE", "8")),
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
