"""The single production path from memory text to an embedding vector."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from memory.embedding_backend import embed_text, embedding_backend_config
from memory.experiment_config import EmbeddingExperimentConfig, configured_backend
from memory.identity_normalization import IdentityNormalizer
from memory.semantic_representation import SemanticMemoryRepresentation, build_semantic_representation

REPRESENTATION_VERSION = "semantic-v1"
IDENTITY_VERSION = "identity-v1"


@dataclass(frozen=True, slots=True)
class EmbeddedText:
    vector: tuple[float, ...]
    embedding_text: str
    representation: SemanticMemoryRepresentation
    metadata: dict[str, Any]


def current_embedding_metadata(experiment: EmbeddingExperimentConfig | None = None) -> dict[str, Any]:
    experiment = experiment or get_embedding_pipeline().config
    config = embedding_backend_config()
    payload = {
        "embedding_backend": config.name,
        "embedding_model": config.model_identifier,
        "embedding_dimension": config.dimensions,
        "representation_version": f"{REPRESENTATION_VERSION}:{experiment.representation_mode}",
        "identity_version": f"{IDENTITY_VERSION}:{'on' if experiment.identity_enabled else 'off'}",
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "backend_fingerprint": fingerprint}


class CanonicalEmbeddingPipeline:
    """Normalize, serialize, and embed production memory/query text."""

    def __init__(self, config: EmbeddingExperimentConfig | None = None) -> None:
        self.config = config or EmbeddingExperimentConfig()
        self.identity_normalizer = IdentityNormalizer()

    def embed_memory(
        self,
        content: str,
        *,
        speaker: str | None = None,
        embedding_text: str | None = None,
    ) -> EmbeddedText:
        source = embedding_text if embedding_text is not None else content
        identity_references: list[str] = []
        if self.config.identity_enabled:
            identity_result = self.identity_normalizer.observe(source, speaker=speaker)
            source = identity_result.text
            identity_references = [f"{alias} -> {canonical}" for alias, canonical in identity_result.replacements.items() if alias != canonical]
        representation = build_semantic_representation(source, speaker=speaker, normalize_identities=False)
        if self.config.representation_mode == "raw":
            serialized = source
        elif self.config.representation_mode == "event":
            from types import SimpleNamespace

            from memory.event_memory.event_builder import EventMemoryBuilder
            from memory.event_memory.event_segmenter import EventSegment
            turn = SimpleNamespace(turn_id="1", speaker=speaker or "user", text=source, session_id="experiment")
            serialized = EventMemoryBuilder().build(EventSegment("experiment", 1, (turn,))).embedding_text
        else:
            serialized = representation.serialize()
        metadata = current_embedding_metadata(self.config)
        vector = tuple(embed_text(serialized, dimensions=metadata["embedding_dimension"]))
        metadata.update(
            {
                "embedding_text": serialized,
                "semantic_representation": representation.to_dict(),
                "identity_references": identity_references,
                "original_content_preserved": True,
            }
        )
        return EmbeddedText(vector, serialized, representation, metadata)

    def create_memory_note(self, content: str, **kwargs: Any) -> Any:
        from memory.memory_note import MemoryNote
        embedded = self.embed_memory(content, speaker=kwargs.pop("speaker", None), embedding_text=kwargs.pop("embedding_text", None))
        note = MemoryNote.create(content=content, embedding=embedded.vector, **kwargs)
        return note.with_updates(retrieval_metadata={**note.retrieval_metadata, **embedded.metadata})

    def embed_query(self, query: str) -> EmbeddedText:
        return self.embed_memory(query)


_PIPELINE = CanonicalEmbeddingPipeline(
    EmbeddingExperimentConfig(
        backend=os.getenv("PRIMA_EMBEDDING_BACKEND", "stable"),
        representation_mode=os.getenv("PRIMA_REPRESENTATION_MODE", "semantic"),
        identity_enabled=os.getenv("PRIMA_IDENTITY_NORMALIZATION", "true").lower() not in {"0", "false", "no", "off"},
    )
)
_ACTIVE_PIPELINE: ContextVar[CanonicalEmbeddingPipeline] = ContextVar("prima_embedding_pipeline", default=_PIPELINE)


def get_embedding_pipeline() -> CanonicalEmbeddingPipeline:
    return _ACTIVE_PIPELINE.get()


@contextmanager
def use_embedding_pipeline(pipeline: CanonicalEmbeddingPipeline) -> Iterator[None]:
    with configured_backend(pipeline.config.backend):
        token = _ACTIVE_PIPELINE.set(pipeline)
        try:
            yield
        finally:
            _ACTIVE_PIPELINE.reset(token)
