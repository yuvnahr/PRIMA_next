"""The single production path from memory text to an embedding vector."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

from memory.embedding_backend import embed_text, embedding_backend_config
from memory.identity_normalization import IdentityNormalizer
from memory.semantic_representation import SemanticMemoryRepresentation, build_semantic_representation
from memory.experiment_config import EmbeddingExperimentConfig, configured_backend

REPRESENTATION_VERSION = "semantic-v1"
IDENTITY_VERSION = "identity-v1"


@dataclass(frozen=True, slots=True)
class EmbeddedText:
    vector: tuple[float, ...]
    embedding_text: str
    representation: SemanticMemoryRepresentation
    metadata: dict[str, Any]


def current_embedding_metadata() -> dict[str, Any]:
    config = embedding_backend_config()
    identity = f"{IDENTITY_VERSION}:on"
    representation = f"{REPRESENTATION_VERSION}:semantic"
    payload = {
        "embedding_backend": config.name,
        "embedding_model": config.model_identifier,
        "embedding_dimension": config.dimensions,
        "representation_version": representation,
        "identity_version": identity,
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
            from memory.event_memory.event_builder import EventMemoryBuilder
            from memory.event_memory.event_segmenter import EventSegment
            from types import SimpleNamespace
            turn = SimpleNamespace(turn_id="1", speaker=speaker or "user", text=source, session_id="experiment")
            serialized = EventMemoryBuilder().build(EventSegment("experiment", 1, (turn,))).embedding_text
        else:
            serialized = representation.serialize()
        metadata = current_embedding_metadata()
        vector = tuple(embed_text(serialized, dimensions=metadata["embedding_dimension"]))
        metadata["representation_version"] = f"{REPRESENTATION_VERSION}:{self.config.representation_mode}"
        metadata["identity_version"] = f"{IDENTITY_VERSION}:{'on' if self.config.identity_enabled else 'off'}"
        fingerprint_payload = {key: metadata[key] for key in ("embedding_backend", "embedding_model", "embedding_dimension", "representation_version", "identity_version")}
        metadata["backend_fingerprint"] = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
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


_PIPELINE = CanonicalEmbeddingPipeline()
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
