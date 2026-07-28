"""Memory note model and backward-compatible ingestion helpers."""

from __future__ import annotations

import importlib.util
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from memory.memory_context import StateSnapshot
from memory.memory_lineage import MemoryLineage
from memory.memory_types import MemoryLevel, MemoryType

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")
GENERIC_NOUNS = {
    "time",
    "day",
    "minute",
    "hour",
    "way",
    "thing",
    "people",
    "lot",
    "today",
    "week",
    "month",
    "year",
    "moment",
}


def stable_embedding(text: str, dimensions: int = 64) -> list[float]:
    """Return the configured embedding while preserving the existing interface."""

    from memory.embedding_backend import embed_text
    return embed_text(text, dimensions)


def tokenize(text: str) -> list[str]:
    return [token.lower().strip("'") for token in TOKEN_RE.findall(text)]


def extract_dominant_context_chain(text: str, top_k: int = 5) -> list[str]:
    """Preserve A-MEM dominant context chain extraction without requiring spaCy."""
    try:
        if importlib.util.find_spec("spacy") is None:
            raise ImportError("spacy unavailable")
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = None
        if nlp is not None:
            doc = nlp(text)
            chunks = []
            for chunk in doc.noun_chunks:
                clean_chunk = " ".join(
                    token.lemma_.lower()
                    for token in chunk
                    if not token.is_stop and token.pos_ in {"NOUN", "PROPN"}
                )
                if clean_chunk and len(clean_chunk.split()) > 1:
                    chunks.append(clean_chunk)
            nouns = []
            for token in doc:
                if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
                    lemma = token.lemma_.lower()
                    if len(lemma) > 2 and lemma not in GENERIC_NOUNS:
                        nouns.append(lemma)
            chunk_words = {word for chunk in chunks for word in chunk.split()}
            filtered_nouns = [noun for noun in nouns if noun not in chunk_words]
            return list(dict.fromkeys(chunks + filtered_nouns))[:top_k]
    except Exception as e:
        if not isinstance(e, ImportError):
            logger.warning("Keyword extraction failed: %s", e)

    words = [word for word in tokenize(text) if len(word) > 2 and word not in GENERIC_NOUNS]
    bigrams = [f"{words[index]} {words[index + 1]}" for index in range(len(words) - 1)]
    return list(dict.fromkeys(bigrams[:2] + words))[:top_k]


def calculate_smart_overlap(query_chain: Sequence[str], memory_keywords: Sequence[str]) -> int:
    """Preserve exact/substring keyword boosting behavior."""
    overlap = 0
    for query_word in set(query_chain):
        for memory_word in memory_keywords:
            if query_word in memory_word or memory_word in query_word:
                overlap += 1
                break
    return overlap


@dataclass(frozen=True, slots=True)
class MemoryNote:
    id: str
    memory_type: MemoryType
    memory_level: MemoryLevel
    content: str
    embedding: tuple[float, ...]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    affective_state: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    lineage: MemoryLineage = field(default_factory=MemoryLineage)
    graph_links: dict[str, Any] = field(default_factory=dict)
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    evolution_metadata: dict[str, Any] = field(default_factory=dict)
    state_snapshot: StateSnapshot = field(default_factory=StateSnapshot)
    salience_score: float = 0.0
    retention_score: float = 1.0
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_type", MemoryType(self.memory_type))
        object.__setattr__(self, "memory_level", MemoryLevel(self.memory_level))
        object.__setattr__(self, "embedding", tuple(float(value) for value in self.embedding))
        object.__setattr__(self, "salience_score", max(0.0, min(1.0, float(self.salience_score))))
        object.__setattr__(self, "retention_score", max(0.0, min(1.0, float(self.retention_score))))

    @property
    def keywords(self) -> list[str]:
        existing = self.retrieval_metadata.get("keywords")
        if isinstance(existing, list):
            return [str(item) for item in existing]
        return extract_dominant_context_chain(self.content)

    def with_updates(self, **updates: Any) -> MemoryNote:
        next_version = int(updates.pop("version", self.version + 1))
        return replace(self, **updates, version=next_version)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "memory_level": self.memory_level.value,
            "timestamp": self.timestamp.isoformat(),
            "timestamp_unix": self.timestamp.timestamp(),
            "source_input": self.content,
            "affective_state": self.affective_state,
            "context": self.context,
            "lineage": self.lineage.to_dict(),
            "graph_links": self.graph_links,
            "retrieval_metadata": self.retrieval_metadata,
            "evolution_metadata": self.evolution_metadata,
            "state_snapshot": self.state_snapshot.to_dict(),
            "salience_score": self.salience_score,
            "retention_score": self.retention_score,
            "version": self.version,
            "keywords": self.keywords,
            "tags": self.retrieval_metadata.get("tags", []),
            **{key: self.retrieval_metadata.get(key) for key in ("embedding_backend", "embedding_model", "embedding_dimension", "representation_version", "identity_version", "backend_fingerprint")},
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document": self.content,
            "embedding": list(self.embedding),
            "metadata": self.to_metadata(),
        }

    @classmethod
    def create(
        cls,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        memory_level: MemoryLevel = MemoryLevel.EPISODIC_EVENT,
        embedding: Sequence[float] | None = None,
        affective_state: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        state_snapshot: StateSnapshot | None = None,
        salience_score: float = 0.0,
        retention_score: float = 1.0,
        note_id: str | None = None,
        embedding_text: str | None = None,
        timestamp: datetime | None = None,
    ) -> MemoryNote:
        from memory.embedding_pipeline import get_embedding_pipeline
        keywords = extract_dominant_context_chain(content)
        embedding_metadata: dict[str, Any] = {}
        if embedding is None:
            embedded = get_embedding_pipeline().embed_memory(content, embedding_text=embedding_text)
            embedding = embedded.vector
            embedding_metadata = embedded.metadata
        return cls(
            id=note_id or f"mem_{uuid.uuid4()}", memory_type=memory_type, memory_level=memory_level,
            content=content, embedding=tuple(embedding), timestamp=timestamp or datetime.now(timezone.utc), affective_state=affective_state or {}, context=context or {},
            retrieval_metadata={"keywords": keywords, "dominant_context_chain": keywords, **embedding_metadata},
            state_snapshot=state_snapshot or StateSnapshot(), salience_score=salience_score, retention_score=retention_score,
        )
    @classmethod
    def from_record(cls, record: dict[str, Any]) -> MemoryNote:
        metadata = record.get("metadata", {})
        timestamp = metadata.get("timestamp")
        parsed_timestamp = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else datetime.now(timezone.utc)
        embedding = record.get("embedding")
        if embedding is None:
            embedding = stable_embedding(str(record.get("document", "")))
        retrieval_metadata = dict(metadata.get("retrieval_metadata", {"keywords": metadata.get("keywords", [])}))
        for key in ("embedding_backend", "embedding_model", "embedding_dimension", "representation_version", "identity_version", "backend_fingerprint"):
            if metadata.get(key) is not None:
                retrieval_metadata.setdefault(key, metadata[key])
        return cls(
            id=str(record.get("id") or metadata.get("id")), memory_type=MemoryType(metadata.get("memory_type", MemoryType.EPISODIC.value)),
            memory_level=MemoryLevel(metadata.get("memory_level", MemoryLevel.EPISODIC_EVENT.value)),
            content=str(record.get("document") or metadata.get("source_input", "")), embedding=tuple(embedding), timestamp=parsed_timestamp,
            affective_state=dict(metadata.get("affective_state", {})), context=dict(metadata.get("context", {})),
            lineage=MemoryLineage.from_dict(metadata.get("lineage")), graph_links=dict(metadata.get("graph_links", {})),
            retrieval_metadata=retrieval_metadata, evolution_metadata=dict(metadata.get("evolution_metadata", {})),
            state_snapshot=StateSnapshot.from_dict(metadata.get("state_snapshot")), salience_score=float(metadata.get("salience_score", 0.0)),
            retention_score=float(metadata.get("retention_score", 1.0)), version=int(metadata.get("version", 1)),
        )

def format_memory_note(input_data: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible A-MEM formatter returning Chroma-style fields."""
    raw_text = str(input_data["raw_text"])
    dominant_emotions = input_data.get("dominant_emotions", {})
    top_emotions = [{"label": emotion, "score": score} for emotion, score in dominant_emotions.items()]
    primary_emotion = top_emotions[0]["label"] if top_emotions else "neutral"
    keywords = extract_dominant_context_chain(raw_text)
    note = MemoryNote.create(
        content=raw_text,
        memory_type=MemoryType.WORKING,
        memory_level=MemoryLevel.RAW,
        embedding=input_data.get("vector_embedding"),
        affective_state={"top_emotions": top_emotions},
        context={"type": "user_input", "source": "bulk_test", "references": [], "meta": {}},
    )
    metadata = note.to_metadata()
    metadata["tags"] = list(dict.fromkeys(["memory_note", primary_emotion]))
    metadata["keywords"] = keywords
    return {"id": note.id, "embedding": list(note.embedding), "metadata": metadata}






