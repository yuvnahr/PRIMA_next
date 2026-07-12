"""The single production path from memory text to an embedding vector."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from memory.embedding_backend import embed_text, embedding_backend_config
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


def current_embedding_metadata() -> dict[str, Any]:
    config = embedding_backend_config()
    identity = IDENTITY_VERSION
    representation = REPRESENTATION_VERSION
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

    def __init__(self) -> None:
        self.identity_normalizer = IdentityNormalizer()

    def embed_memory(
        self,
        content: str,
        *,
        speaker: str | None = None,
        embedding_text: str | None = None,
    ) -> EmbeddedText:
        representation = build_semantic_representation(
            embedding_text if embedding_text is not None else content,
            speaker=speaker,
            identity_normalizer=self.identity_normalizer,
            normalize_identities=True,
        )
        serialized = representation.serialize()
        metadata = current_embedding_metadata()
        metadata.update(
            {
                "embedding_text": serialized,
                "semantic_representation": representation.to_dict(),
                "original_content_preserved": True,
            }
        )
        return EmbeddedText(tuple(embed_text(serialized, dimensions=metadata["embedding_dimension"])), serialized, representation, metadata)

    def embed_query(self, query: str) -> EmbeddedText:
        return self.embed_memory(query)


_PIPELINE = CanonicalEmbeddingPipeline()


def get_embedding_pipeline() -> CanonicalEmbeddingPipeline:
    return _PIPELINE
