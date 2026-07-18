"""Repository layer for Chroma-backed or deterministic in-memory storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import os
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List

import numpy as np

from memory.embedding_pipeline import current_embedding_metadata
from memory.memory_metadata import decode_metadata, encode_metadata
from memory.memory_note import MemoryNote
from memory.memory_types import COLLECTION_BY_TYPE, MemoryType


class MemoryRepository(ABC):
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
    def list(self, memory_type: MemoryType | None = None) -> List[MemoryNote]:
        raise NotImplementedError

    @abstractmethod
    def query(self, embedding: Iterable[float], memory_type: MemoryType | None = None, limit: int = 10) -> List[tuple[MemoryNote, float]]:
        raise NotImplementedError


class InMemoryMemoryRepository(MemoryRepository):
    """Deterministic repository used by tests and offline development."""

    def __init__(self) -> None:
        self._notes: dict[MemoryType, dict[str, MemoryNote]] = defaultdict(dict)
        self.fingerprint_status = "valid"

    def add(self, note: MemoryNote) -> MemoryNote:
        self._notes[note.memory_type][note.id] = note
        return note

    def update(self, note: MemoryNote) -> MemoryNote:
        self._notes[note.memory_type][note.id] = note
        return note

    def get(self, note_id: str, memory_type: MemoryType | None = None) -> MemoryNote | None:
        if memory_type is not None:
            return self._notes[memory_type].get(note_id)
        for notes in self._notes.values():
            if note_id in notes:
                return notes[note_id]
        return None

    def list(self, memory_type: MemoryType | None = None) -> List[MemoryNote]:
        if memory_type is not None:
            return list(self._notes[memory_type].values())
        notes: list[MemoryNote] = []
        for typed_notes in self._notes.values():
            notes.extend(typed_notes.values())
        return notes

    def query(self, embedding: Iterable[float], memory_type: MemoryType | None = None, limit: int = 10) -> List[tuple[MemoryNote, float]]:
        query = np.array(list(embedding), dtype="float32")
        query_norm = np.linalg.norm(query) or 1.0
        query = query / query_norm
        scored: list[tuple[MemoryNote, float]] = []
        for note in self.list(memory_type):
            vector = np.array(note.embedding, dtype="float32")
            vector_norm = np.linalg.norm(vector) or 1.0
            score = float(np.dot(query, vector / vector_norm))
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
        self.client: Any = chromadb.PersistentClient(path=path)
        self.collections: dict[MemoryType, Any] = {
            memory_type: self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            for memory_type, collection_name in COLLECTION_BY_TYPE.items()
        }
        self.fingerprint_status = self._validate_fingerprint()

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
        raise RuntimeError(f"Embedding fingerprint mismatch: stored={sorted(str(item) for item in found)} current={expected}")

    def _read_repository_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def _write_repository_metadata(self) -> None:
        self.repository_metadata = current_embedding_metadata()
        self.metadata_path.write_text(json.dumps(self.repository_metadata, sort_keys=True), encoding="utf-8")
    def add(self, note: MemoryNote) -> MemoryNote:
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
        memory_types = [memory_type] if memory_type else list(MemoryType)
        for candidate_type in memory_types:
            collection = self.collections[candidate_type]
            result = collection.get(ids=[note_id], include=["embeddings", "documents", "metadatas"])
            if result["ids"]:
                return self._from_chroma_result(result, 0)
        return None

    def list(self, memory_type: MemoryType | None = None) -> List[MemoryNote]:
        notes: list[MemoryNote] = []
        memory_types = [memory_type] if memory_type else list(MemoryType)
        for candidate_type in memory_types:
            result = self.collections[candidate_type].get(include=["embeddings", "documents", "metadatas"])
            notes.extend(self._from_chroma_result(result, index) for index in range(len(result["ids"])))
        return notes

    def query(self, embedding: Iterable[float], memory_type: MemoryType | None = None, limit: int = 10) -> List[tuple[MemoryNote, float]]:
        if self.fingerprint_status != "valid":
            raise RuntimeError("Repository fingerprint is invalid; rebuild before retrieval")
        notes: List[tuple[MemoryNote, float]] = []
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


