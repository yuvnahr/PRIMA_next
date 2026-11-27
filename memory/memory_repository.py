"""Repository layer for Chroma-backed or deterministic in-memory storage."""

from __future__ import annotations

import builtins
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, cast

import numpy as np
from filelock import FileLock
from numpy.typing import NDArray

from memory.embedding_backend import embedding_backend_config
from memory.embedding_pipeline import current_embedding_metadata
from memory.memory_metadata import decode_metadata, encode_metadata
from memory.memory_note import MemoryNote
from memory.memory_types import COLLECTION_BY_TYPE, MemoryType

_STORE_LOCK = RLock()


def _move_record(note: MemoryNote) -> dict[str, Any]:
    record = note.to_record()
    record["embedding"] = np.asarray(note.embedding, dtype=np.float32).tolist()
    return json.loads(json.dumps(record, allow_nan=False))


def validate_vector(note: MemoryNote, identity: dict[str, Any]) -> None:
    vector = np.asarray(note.embedding, dtype=np.float32)
    if vector.shape != (identity["embedding_dimension"],) or not np.isfinite(vector).all() or not np.any(vector):
        raise ValueError("Incompatible embedding dimension or nonfinite/zero vector.")
    identity_fields = (
        "backend_fingerprint",
        "embedding_backend",
        "embedding_model",
        "embedding_dimension",
        "representation_version",
        "identity_version",
    )
    if any(note.retrieval_metadata.get(key) != identity[key] for key in identity_fields):
        raise ValueError("Stored row embedding fingerprint mismatch.")


class MemoryRepository(ABC):
    @contextmanager
    def transaction(self):
        """Serialize local readers and writers, including snapshot-wide maintenance."""
        with _STORE_LOCK:
            yield

    def move(self, source: MemoryNote, target: MemoryNote) -> MemoryNote:
        raise NotImplementedError

    def delete(self, note_id: str, memory_type: MemoryType) -> None:
        raise NotImplementedError

    def _check_move(self, source: MemoryNote, target: MemoryNote) -> bool:
        if source.id != target.id or source.memory_type == target.memory_type:
            raise ValueError("A move must retain the ID and change the type.")
        old = self.get(source.id, source.memory_type)
        existing = self.get(target.id, target.memory_type)
        if existing is not None:
            if old is None and _move_record(existing) == _move_record(target):
                return False
            raise ValueError("Conflicting move destination.")
        if old is None or _move_record(old) != _move_record(source):
            raise ValueError("Stale move source.")
        return True

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
        self.published_identity: dict[str, Any] | None = None
        self.previous_generation: (
            tuple[dict[MemoryType, dict[str, MemoryNote]], str, str, dict[str, Any] | None] | None
        ) = None

    def _check_generation(self, note: MemoryNote | None = None) -> None:
        if self.published_identity is None:
            return
        if self.published_identity["backend_fingerprint"] != current_embedding_metadata()["backend_fingerprint"]:
            raise RuntimeError("Stale embedding writer; select the active configuration.")
        if note is not None:
            validate_vector(note, self.published_identity)

    def add(self, note: MemoryNote) -> MemoryNote:
        with self.transaction():
            self._check_generation(note)
            if any(note.id in rows for kind, rows in self._notes.items() if kind != note.memory_type):
                raise ValueError("ID already belongs to another memory type; use move.")
            previous = self._notes[note.memory_type].get(note.id)
            if (
                previous is None
                or previous.embedding != note.embedding
                or previous.retrieval_metadata.get("backend_fingerprint")
                != note.retrieval_metadata.get("backend_fingerprint")
            ):
                self._normalized_vectors.pop((note.memory_type, note.id), None)
            self._notes[note.memory_type][note.id] = note
            return note

    def move(self, source: MemoryNote, target: MemoryNote) -> MemoryNote:
        with self.transaction():
            self._check_generation(target)
            if self._check_move(source, target):
                self._notes[target.memory_type][target.id] = target
                self.delete(source.id, source.memory_type)
            return target

    def delete(self, note_id: str, memory_type: MemoryType) -> None:
        with self.transaction():
            self._check_generation()
            self._notes[memory_type].pop(note_id, None)
            self._normalized_vectors.pop((memory_type, note_id), None)

    def update(self, note: MemoryNote) -> MemoryNote:
        with self.transaction():
            return self.add(note)

    def get(self, note_id: str, memory_type: MemoryType | None = None) -> MemoryNote | None:
        with self.transaction():
            if memory_type is not None:
                return self._notes[memory_type].get(note_id)
            for notes in self._notes.values():
                if note_id in notes:
                    return notes[note_id]
            return None

    def list(self, memory_type: MemoryType | None = None) -> builtins.list[MemoryNote]:
        with self.transaction():
            if memory_type is not None:
                return list(self._notes[memory_type].values())
            notes: list[MemoryNote] = []
            for typed_notes in self._notes.values():
                notes.extend(typed_notes.values())
            return notes

    def query(self, embedding: Iterable[float], memory_type: MemoryType | None = None, limit: int = 10) -> builtins.list[tuple[MemoryNote, float]]:
        with self.transaction():
            query = np.array(list(embedding), dtype="float32")
            if query.ndim != 1 or not query.size or not np.isfinite(query).all():
                raise ValueError("Query embedding must be a finite nonempty vector")
            notes = self.list(memory_type)
            learned = embedding_backend_config().name != "stable"
            expected = (
                current_embedding_metadata()["backend_fingerprint"]
                if learned or any(note.retrieval_metadata.get("backend_fingerprint") for note in notes)
                else None
            )
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

    def __init__(self, path: str = "./memory_db", *, inspection: bool = False) -> None:
        import chromadb

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.path / ".prima_embedding_metadata.json"
        self.file_lock = FileLock(str(self.path.resolve() / ".prima_store.lock"), timeout=5, is_singleton=True)
        self.move_path = self.path / ".prima_pending_move.json"
        self.inspection = inspection
        with _STORE_LOCK, self.file_lock:
            self.repository_metadata = self._read_repository_metadata()
            self.client: Any = chromadb.PersistentClient(path=str(self.path))
            self.generation = self.repository_metadata.get("generation", "legacy")
            if self.repository_metadata:
                self.collections = self.open_collections(self.generation)
            else:
                if inspection:
                    raise ValueError("Cannot inspect existing store without an active embedding manifest.")
                existing = {getattr(c, "name", str(c)) for c in self.client.list_collections()}
                if any(name.startswith("prima_gen_") for name in existing) or any(
                    self.client.get_collection(name=name).count()
                    for name in existing & set(COLLECTION_BY_TYPE.values())
                ):
                    raise ValueError("Missing active manifest for nonempty content store.")
                self.collections = {
                    kind: self.client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
                    for kind, name in COLLECTION_BY_TYPE.items()
                }
                self.publish_manifest({**current_embedding_metadata(), "generation": "legacy", "format_version": 1})
                self.repository_metadata = self._read_repository_metadata()
            if self.move_path.exists():
                if inspection:
                    raise RuntimeError("Pending promotion; reopen normally to recover before migration.")
                self._finish_move()
            seen: set[str] = set()
            for collection in self.collections.values():
                ids = set(collection.get(include=[])["ids"])
                if seen & ids:
                    raise ValueError("Duplicate memory owners in legacy store; reconcile a backup offline.")
                seen.update(ids)
            self.fingerprint_status = self._validate_fingerprint()

    def open_collections(self, generation: str, *, create: bool = False) -> dict[MemoryType, Any]:
        collections = {}
        for kind, legacy in COLLECTION_BY_TYPE.items():
            name = legacy if generation == "legacy" else f"prima_gen_{generation}_{legacy}"
            collections[kind] = (
                self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
                if create
                else self.client.get_collection(name=name)
            )
        return collections

    def publish_manifest(self, manifest: dict[str, Any]) -> None:
        temp = self.metadata_path.with_suffix(".tmp")
        temp.write_text(json.dumps(manifest, sort_keys=True, allow_nan=False), encoding="utf-8")
        temp.replace(self.metadata_path)

    @contextmanager
    def transaction(self):
        with _STORE_LOCK, self.file_lock:
            active = self._read_repository_metadata()
            if not active or active.get("generation", "legacy") != self.generation:
                raise RuntimeError("Active embedding generation changed; reopen repository.")
            if self.move_path.exists():
                self._finish_move()
            yield

    def move(self, source: MemoryNote, target: MemoryNote) -> MemoryNote:
        with self.transaction():
            self._require_writable()
            validate_vector(target, self.repository_metadata)
            if self._check_move(source, target):
                intent = {"source": _move_record(source), "target": _move_record(target)}
                temp = self.move_path.with_suffix(".tmp")
                temp.write_text(json.dumps(intent, allow_nan=False), encoding="utf-8")
                temp.replace(self.move_path)
                self._finish_move()
            return target

    def _finish_move(self) -> None:
        intent = json.loads(self.move_path.read_text(encoding="utf-8"))
        source, target = (MemoryNote.from_record(intent[key]) for key in ("source", "target"))
        old = self._read_note(source.id, source.memory_type)
        if old is not None and _move_record(old) != intent["source"]:
            raise ValueError("Move source changed; reconcile pending intent.")
        copied = self._read_note(target.id, target.memory_type)
        if copied is None:
            if old is None:
                raise ValueError("Both move owners missing.")
            self.collections[target.memory_type].upsert(
                ids=[target.id],
                documents=[target.content],
                embeddings=[list(target.embedding)],
                metadatas=[encode_metadata(target.to_metadata())],
            )
            copied = self._read_note(target.id, target.memory_type)
        if copied is None or _move_record(copied) != intent["target"]:
            raise ValueError("Move destination does not match intent.")
        if old is not None:
            self.collections[source.memory_type].delete(ids=[source.id])
        self.move_path.unlink()

    def delete(self, note_id: str, memory_type: MemoryType) -> None:
        with self.transaction():
            self._require_writable()
            self.collections[memory_type].delete(ids=[note_id])

    def _validate_fingerprint(self) -> str:
        expected = current_embedding_metadata()["backend_fingerprint"]
        found = self.repository_metadata["backend_fingerprint"]
        for note in self.list():
            validate_vector(note, self.repository_metadata)
        if found == expected:
            return "valid"
        return self._mismatch_status(found, expected)

    def _mismatch_status(self, found: object, expected: str) -> str:
        if self.inspection:
            return "readonly"
        mode = os.getenv("PRIMA_EMBEDDING_MISMATCH", "error").lower()
        if mode == "readonly":
            return "readonly"
        if mode == "rebuild":
            return "invalid-rebuild-required"
        raise RuntimeError(f"Embedding fingerprint mismatch: stored={found} current={expected}")

    def _read_repository_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not {"backend_fingerprint", "embedding_dimension"} <= payload.keys():
            raise ValueError("Invalid active embedding manifest.")
        if "format_version" in payload and (payload["format_version"] != 1 or not payload.get("generation")):
            raise ValueError("Invalid active embedding generation.")
        return payload

    def _require_writable(self) -> None:
        if (
            self.inspection
            or self.fingerprint_status != "valid"
            or current_embedding_metadata()["backend_fingerprint"] != self.repository_metadata["backend_fingerprint"]
        ):
            raise RuntimeError("Incompatible embedding generation; reopen or rebuild before writing.")

    def add(self, note: MemoryNote) -> MemoryNote:
        with self.transaction():
            self._require_writable()
            validate_vector(note, self.repository_metadata)
            if any(
                col.get(ids=[note.id], include=[])["ids"]
                for kind, col in self.collections.items()
                if kind != note.memory_type
            ):
                raise ValueError("ID already belongs to another memory type; use move.")
            collection: Any = self.collections[note.memory_type]
            collection.add(
                ids=[note.id],
                documents=[note.content],
                embeddings=[list(note.embedding)],
                metadatas=[encode_metadata(note.to_metadata())],
            )
            return note

    def update(self, note: MemoryNote) -> MemoryNote:
        with self.transaction():
            self._require_writable()
            validate_vector(note, self.repository_metadata)
            if any(
                col.get(ids=[note.id], include=[])["ids"]
                for kind, col in self.collections.items()
                if kind != note.memory_type
            ):
                raise ValueError("ID already belongs to another memory type; use move.")
            collection: Any = self.collections[note.memory_type]
            collection.upsert(
                ids=[note.id],
                documents=[note.content],
                embeddings=[list(note.embedding)],
                metadatas=[encode_metadata(note.to_metadata())],
            )
            return note

    def get(self, note_id: str, memory_type: MemoryType | None = None) -> MemoryNote | None:
        with self.transaction():
            return self._read_note(note_id, memory_type)

    def _read_note(self, note_id: str, memory_type: MemoryType | None) -> MemoryNote | None:
        memory_types = [memory_type] if memory_type else list(MemoryType)
        for candidate_type in memory_types:
            collection = self.collections[candidate_type]
            result = collection.get(ids=[note_id], include=["embeddings", "documents", "metadatas"])
            if result["ids"]:
                return self._from_chroma_result(result, 0)
        return None

    def list(self, memory_type: MemoryType | None = None) -> builtins.list[MemoryNote]:
        with self.transaction():
            notes: list[MemoryNote] = []
            memory_types = [memory_type] if memory_type else list(MemoryType)
            for candidate_type in memory_types:
                result = self.collections[candidate_type].get(include=["embeddings", "documents", "metadatas"])
                notes.extend(self._from_chroma_result(result, index) for index in range(len(result["ids"])))
            return notes

    def query(self, embedding: Iterable[float], memory_type: MemoryType | None = None, limit: int = 10) -> builtins.list[tuple[MemoryNote, float]]:
        with self.transaction():
            if (
                self.fingerprint_status != "valid"
                or current_embedding_metadata()["backend_fingerprint"]
                != self.repository_metadata["backend_fingerprint"]
            ):
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
