"""The single production path from memory text to an embedding vector."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

from memory.embedding_backend import (
    EmbeddingBackendConfig,
    embed_text,
    embedding_backend_config,
    get_embedding_backend,
    use_embedding_backend_config,
)
from memory.experiment_config import EmbeddingExperimentConfig
from memory.identity_normalization import IdentityNormalizer
from memory.semantic_representation import (
    SemanticMemoryRepresentation,
    build_semantic_representation,
)

REPRESENTATION_VERSION = "semantic-v1"
IDENTITY_VERSION = "identity-v2"


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
        "identity_version": f"{IDENTITY_VERSION}:on" if experiment.identity_enabled else "identity-v1:off",
    }
    if config.name != "stable":
        info = get_embedding_backend().info()
        payload.update(embedding_revision=config.revision, embedding_device=config.device,
                       embedding_batch_size=config.batch_size, embedding_cpu_threads=config.cpu_threads,
                       embedding_snapshot_sha256=(info["snapshot"] or {}).get("artifact_sha256"), normalization="l2-float32")
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
        return self._embed(content, speaker, embedding_text, normalize=self.config.identity_enabled)

    def _embed(self, content: str, speaker: str | None, embedding_text: str | None, *, normalize: bool) -> EmbeddedText:
        source = embedding_text if embedding_text is not None else content
        identity_references: list[str] = []
        if normalize:
            identity_result = IdentityNormalizer().observe(source, speaker=speaker)
            source = identity_result.text
            identity_references = [f"{alias} -> {canonical}" for alias, canonical in identity_result.replacements.items() if alias != canonical]
        representation = build_semantic_representation(source, speaker=speaker, normalize_identities=False)
        representation = replace(representation, original_text=embedding_text if embedding_text is not None else content)
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
                **({"embedding_source_text": embedding_text} if embedding_text is not None else {}),
                **({"embedding_speaker": speaker} if speaker is not None else {}),
                "embedding_text": serialized,
                "semantic_representation": representation.to_dict(),
                "identity_references": identity_references,
                "original_content_preserved": True,
            }
        )
        return EmbeddedText(vector, serialized, representation, metadata)

    @staticmethod
    def stored_embedding_source(note: Any) -> str | None:
        source = note.retrieval_metadata.get("embedding_source_text")
        if source is None and note.context.get("type") == "event_memory":
            source = note.retrieval_metadata.get("embedding_text")
        if source is None:
            original = note.retrieval_metadata.get("semantic_representation", {}).get("original_text")
            if original and original != note.content:
                reconstructed = note.content
                for declaration in note.retrieval_metadata.get("identity_references", []):
                    alias, separator, name = declaration.partition(" -> ")
                    if separator:
                        reconstructed = re.sub(rf"\b{re.escape(alias)}\b", name.replace("\\", "\\\\"), reconstructed)
                if original != reconstructed:
                    raise ValueError("Ambiguous legacy embedding input; supply reviewed embedding_source_text.")
        return source

    def embed_stored_note(self, note: Any) -> EmbeddedText:
        source = self.stored_embedding_source(note)
        return self.embed_memory(
            note.content,
            speaker=note.retrieval_metadata.get("embedding_speaker", note.context.get("speaker")),
            embedding_text=source,
        )

    def create_memory_note(self, content: str, **kwargs: Any) -> Any:
        from memory.memory_note import MemoryNote
        embedded = self.embed_memory(content, speaker=kwargs.pop("speaker", None), embedding_text=kwargs.pop("embedding_text", None))
        note = MemoryNote.create(content=content, embedding=embedded.vector, **kwargs)
        return note.with_updates(retrieval_metadata={**note.retrieval_metadata, **embedded.metadata})

    def embed_query(self, query: str) -> EmbeddedText:
        return self._embed(query, None, None, normalize=False)


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
def use_embedding_pipeline(
    pipeline: CanonicalEmbeddingPipeline,
    backend_config: EmbeddingBackendConfig | None = None,
) -> Iterator[None]:
    resolved = backend_config or embedding_backend_config()
    with use_embedding_backend_config(resolved):
        token = _ACTIVE_PIPELINE.set(pipeline)
        try:
            yield
        finally:
            _ACTIVE_PIPELINE.reset(token)
