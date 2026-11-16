"""Repository layer for Chroma-backed or deterministic in-memory storage."""

from __future__ import annotations

import builtins
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from threading import RLock
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from memory.embedding_backend import embedding_backend_config
from memory.embedding_pipeline import current_embedding_metadata
from memory.memory_metadata import configure_collection_sync, decode_metadata, encode_metadata
from memory.memory_note import MemoryNote
from memory.memory_types import COLLECTION_BY_TYPE, MemoryType


class MemoryRepository(ABC):
    def count(self) -> int:
        return len(self.list())

    def close(self) -> None:
        """Release owned resources when the backend exposes an explicit lifecycle."""
        return None

    @abstractmethod
    def clear(self) -> None:
        """Clear this repository's contents only."""
        raise NotImplementedError

    @abstractmethod
    def add(self, note: MemoryNote) -> MemoryNote:
        raise NotImplementedError

    @abstractmethod
    def update(self, note: MemoryNote) -> MemoryNote:
        raise NotImplementedError

    @abstractmethod
    def get(self, note_id: str, memory_type: MemoryType | None = None) -> MemoryNote | None:
        raise NotImplementedError

    @abstractmethod
    def list(self, memory_type: MemoryType | None = None) -> builtins.list[MemoryNote]:
        raise NotImplementedError

    @abstractmethod
    def query(self, embedding: Iterable[float], memory_type: MemoryType | None = None, limit: int = 10) -> builtins.list[tuple[MemoryNote, float]]:
        raise NotImplementedError


class InMemoryMemoryRepository(MemoryRepository):
    """Deterministic repository used by tests and offline development."""

    def __init__(self) -> None:
        self._notes: dict[MemoryType, dict[str, MemoryNote]] = defaultdict(dict)
        self._normalized_vectors: dict[tuple[MemoryType, str], NDArray[np.float32]] = {}
        self.fingerprint_status = "valid"

    def add(self, note: MemoryNote) -> MemoryNote:
        previous = self._notes[note.memory_type].get(note.id)
        if previous is None or previous.embedding != note.embedding or previous.retrieval_metadata.get("backend_fingerprint") != note.retrieval_metadata.get("backend_fingerprint"):
            self._normalized_vectors.pop((note.memory_type, note.id), None)
        self._notes[note.memory_type][note.id] = note
        return note

    def clear(self) -> None:
        self._notes.clear()

    def update(self, note: MemoryNote) -> MemoryNote:
        return self.add(note)

    def get(self, note_id: str, memory_type: MemoryType | None = None) -> MemoryNote | None:
        if memory_type is not None:
            return self._notes[memory_type].get(note_id)
        for notes in self._notes.values():
            if note_id in notes:
                return notes[note_id]
        return None

    def list(self, memory_type: MemoryType | None = None) -> builtins.list[MemoryNote]:
        if memory_type is not None:
            return list(self._notes[memory_type].values())
        notes: list[MemoryNote] = []
        for typed_notes in self._notes.values():
            notes.extend(typed_notes.values())
        return notes

    def query(self, embedding: Iterable[float], memory_type: MemoryType | None = None, limit: int = 10) -> builtins.list[tuple[MemoryNote, float]]:
        query = np.array(list(embedding), dtype="float32")
        if query.ndim != 1 or not query.size or not np.isfinite(query).all():
            raise ValueError("Query embedding must be a finite nonempty vector")
        notes = self.list(memory_type)
        learned = embedding_backend_config().name != "stable"
        expected = current_embedding_metadata()["backend_fingerprint"] if learned or any(note.retrieval_metadata.get("backend_fingerprint") for note in notes) else None
        for note in notes:
            found = note.retrieval_metadata.get("backend_fingerprint")
            if len(note.embedding) != len(query) or not np.isfinite(note.embedding).all():
                raise ValueError("Stored embedding dimension/values are incompatible; rebuild the in-memory index")
            if (found is not None and found != expected) or (learned and found is None):
                raise RuntimeError("Embedding fingerprint mismatch; rebuild the in-memory index before retrieval")
        query_norm = np.linalg.norm(query) or 1.0
        query = query / query_norm
        scored: list[tuple[MemoryNote, float]] = []
        for note in notes:
            key = (note.memory_type, note.id)
            vector = self._normalized_vectors.get(key)
            if vector is None:
                raw = np.array(note.embedding, dtype="float32")
                vector = cast(NDArray[np.float32], raw / (np.linalg.norm(raw) or 1.0))
                self._normalized_vectors[key] = vector
            score = float(np.dot(query, vector))
            scored.append((note, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


class ChromaMemoryRepository(MemoryRepository):
    """ChromaDB repository. Chroma remains the intended source of truth."""

    def __init__(self, path: str = "./memory_db") -> None:
        import chromadb

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.path / ".prima_embedding_metadata.json"
        self.repository_metadata = self._read_repository_metadata()
        # Serialize hot-path access with cold maintenance writes, including metadata updates.
        self._lock = RLock()
        self.client: Any = chromadb.PersistentClient(path=path)
        self._closed = False
        try:
            self.collections: dict[MemoryType, Any] = {
                memory_type: self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                for memory_type, collection_name in COLLECTION_BY_TYPE.items()
            }
            for collection in self.collections.values():
                configure_collection_sync(collection)
            self.fingerprint_status = self._validate_fingerprint()
        except BaseException:
            self.close()
            raise

    def count(self) -> int:
        with self._lock:
            return sum(collection.count() for collection in self.collections.values())

    def clear(self) -> None:
        # Recreate collections so an empty store no longer retains old vector dimensions.
        with self._lock:
            for memory_type, name in COLLECTION_BY_TYPE.items():
                self.client.delete_collection(name=name)
                self.collections[memory_type] = self.client.get_or_create_collection(
                    name=name, metadata={"hnsw:space": "cosine"},
                )
                configure_collection_sync(self.collections[memory_type])
            self.metadata_path.unlink(missing_ok=True)
            self.repository_metadata = {}
            self.fingerprint_status = "valid"

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                close = getattr(self.client, "close", None)
                if callable(close):
                    close()

    def _validate_fingerprint(self) -> str:
        expected = current_embedding_metadata()["backend_fingerprint"]
        if self.repository_metadata:
            found = self.repository_metadata.get("backend_fingerprint")
            if found == expected:
                return "valid"
            return self._mismatch_status(found, expected)
        notes = self.list()
        if not notes:
            return "valid"
        found = {note.retrieval_metadata.get("backend_fingerprint") for note in notes}
        if found == {expected}:
            return "valid"
        return self._mismatch_status(found, expected)

    def _mismatch_status(self, found: object, expected: str) -> str:
        mode = os.getenv("PRIMA_EMBEDDING_MISMATCH", "error").lower()
        if mode == "readonly":
            return "readonly"
        if mode == "rebuild":
            return "invalid-rebuild-required"
        raise RuntimeError(f"Embedding fingerprint mismatch: stored={found!r} current={expected}")

    def _read_repository_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def _write_repository_metadata(self) -> None:
        self.repository_metadata = current_embedding_metadata()
        self.metadata_path.write_text(json.dumps(self.repository_metadata, sort_keys=True), encoding="utf-8")
    def add(self, note: MemoryNote) -> MemoryNote:
        with self._lock:
            if self.fingerprint_status == "readonly":
                raise RuntimeError("Repository is readonly because embedding fingerprints differ")
            collection: Any = self.collections[note.memory_type]
            collection.add(
                ids=[note.id],
                documents=[note.content],
                embeddings=[list(note.embedding)],
                metadatas=[encode_metadata(note.to_metadata())],
            )
            self._write_repository_metadata()
            return note

    def update(self, note: MemoryNote) -> MemoryNote:
        with self._lock:
            if self.fingerprint_status == "readonly":
                raise RuntimeError("Repository is readonly because embedding fingerprints differ")
            collection: Any = self.collections[note.memory_type]
            collection.upsert(
                ids=[note.id],
                documents=[note.content],
                embeddings=[list(note.embedding)],
                metadatas=[encode_metadata(note.to_metadata())],
            )
            self._write_repository_metadata()
            return note

    def get(self, note_id: str, memory_type: MemoryType | None = None) -> MemoryNote | None:
        with self._lock:
            memory_types = [memory_type] if memory_type else list(MemoryType)
            for candidate_type in memory_types:
                collection = self.collections[candidate_type]
                result = collection.get(ids=[note_id], include=["embeddings", "documents", "metadatas"])
                if result["ids"]:
                    return self._from_chroma_result(result, 0)
            return None

    def list(self, memory_type: MemoryType | None = None) -> builtins.list[MemoryNote]:
        with self._lock:
            notes: list[MemoryNote] = []
            memory_types = [memory_type] if memory_type else list(MemoryType)
            for candidate_type in memory_types:
                result = self.collections[candidate_type].get(include=["embeddings", "documents", "metadatas"])
                notes.extend(self._from_chroma_result(result, index) for index in range(len(result["ids"])))
            return notes

    def query(self, embedding: Iterable[float], memory_type: MemoryType | None = None, limit: int = 10) -> builtins.list[tuple[MemoryNote, float]]:
        with self._lock:
            if self.fingerprint_status != "valid":
                raise RuntimeError("Repository fingerprint is invalid; rebuild before retrieval")
            notes: list[tuple[MemoryNote, float]] = []
            memory_types = [memory_type] if memory_type else list(MemoryType)
            for candidate_type in memory_types:
                result = self.collections[candidate_type].query(
                    query_embeddings=[list(embedding)],
                    n_results=limit,
                    include=["embeddings", "documents", "metadatas", "distances"],
                )
                ids = result.get("ids", [[]])[0]
                for index in range(len(ids)):
                    record = {
                        "id": ids[index],
                        "document": result["documents"][0][index],
                        "embedding": result["embeddings"][0][index],
                        "metadata": decode_metadata(result["metadatas"][0][index]),
                    }
                    distance = float(result["distances"][0][index])
                    notes.append((MemoryNote.from_record(record), 1.0 - distance))
            return sorted(notes, key=lambda item: item[1], reverse=True)[:limit]

    def _from_chroma_result(self, result: dict, index: int) -> MemoryNote:
        record = {
            "id": result["ids"][index],
            "document": result["documents"][index],
            "embedding": result["embeddings"][index],
            "metadata": decode_metadata(result["metadatas"][index]),
        }
        return MemoryNote.from_record(record)


