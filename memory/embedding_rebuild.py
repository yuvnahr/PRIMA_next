"""Stage, validate and publish a complete embedding generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from memory.embedding_pipeline import current_embedding_metadata, get_embedding_pipeline
from memory.memory_index import MemoryIndex
from memory.memory_metadata import encode_metadata
from memory.memory_note import MemoryNote
from memory.memory_repository import (
    ChromaMemoryRepository,
    InMemoryMemoryRepository,
    MemoryRepository,
    _move_record,
    validate_vector,
)


def _digest(rows: list[MemoryNote]) -> str:
    records = [_move_record(row) for row in sorted(rows, key=lambda r: (r.memory_type.value, r.id))]
    return hashlib.sha256(json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read_generation(repo: ChromaMemoryRepository, generation: str) -> list[MemoryNote]:
    notes = []
    for kind, collection in repo.open_collections(generation).items():
        data = collection.get(include=["embeddings", "documents", "metadatas"])
        for index in range(len(data["ids"])):
            row = repo._from_chroma_result(data, index)
            if row.memory_type != kind:
                raise ValueError("Row in incorrect typed collection.")
            notes.append(row)
    return notes


def rebuild_embeddings(repository: MemoryRepository) -> int:
    """Re-embed authoritative input into a new collection set; return its validated row count."""
    if isinstance(repository, MemoryIndex):
        return rebuild_embeddings(repository.repository)
    if not isinstance(repository, (InMemoryMemoryRepository, ChromaMemoryRepository)):
        raise TypeError("Repository does not implement atomic embedding publication.")
    with repository.transaction():
        source = repository.list()
        identity = current_embedding_metadata()
        generation = uuid.uuid4().hex
        collections = (
            repository.open_collections(generation, create=True)
            if isinstance(repository, ChromaMemoryRepository)
            else {}
        )
        staged = InMemoryMemoryRepository()
        for note in source:
            embedded = get_embedding_pipeline().embed_stored_note(note)
            row = replace(
                note, embedding=embedded.vector, retrieval_metadata={**note.retrieval_metadata, **embedded.metadata}
            )
            validate_vector(row, identity)
            staged.add(row)
            if collections:
                collections[row.memory_type].add(
                    ids=[row.id],
                    documents=[row.content],
                    embeddings=[list(row.embedding)],
                    metadatas=[encode_metadata(row.to_metadata())],
                )
        expected = _digest(staged.list())
        if isinstance(repository, InMemoryMemoryRepository):
            repository.previous_generation = (
                repository._notes,
                expected,
                _digest(source),
                repository.published_identity,
            )
            repository.published_identity = identity
            repository._notes = staged._notes
            repository._normalized_vectors.clear()
        else:
            if _digest(_read_generation(repository, generation)) != expected:
                raise ValueError("Staged generation failed row validation.")
            prior = {
                k: v
                for k, v in repository.repository_metadata.items()
                if k not in {"previous", "published_digest", "previous_digest"}
            }
            manifest = {
                **identity,
                "format_version": 1,
                "generation": generation,
                "previous": prior,
                "previous_digest": _digest(source),
                "published_digest": expected,
            }
            repository.publish_manifest(manifest)
            repository.collections = collections
            repository.generation = generation
            repository.repository_metadata = manifest
            repository.fingerprint_status = "valid"
        return len(source)


def rollback_embeddings(repository: ChromaMemoryRepository | InMemoryMemoryRepository) -> None:
    """Restore an intact prior generation only while the published content still matches."""
    with repository.transaction():
        current = _digest(repository.list())
        if isinstance(repository, InMemoryMemoryRepository):
            prior = repository.previous_generation
            if prior is None or current != prior[1]:
                raise ValueError("No unchanged generation to roll back.")
            if _digest([row for typed in prior[0].values() for row in typed.values()]) != prior[2]:
                raise ValueError("Prior generation failed validation.")
            repository._notes = prior[0]
            repository.published_identity = prior[3]
            repository._normalized_vectors.clear()
            repository.previous_generation = None
            return
        manifest = repository.repository_metadata
        previous = manifest.get("previous")
        if not previous or current != manifest.get("published_digest"):
            raise ValueError("No unchanged generation to roll back.")
        generation = previous.get("generation", "legacy")
        rows = _read_generation(repository, generation)
        if _digest(rows) != manifest["previous_digest"]:
            raise ValueError("Prior generation failed validation.")
        for row in rows:
            validate_vector(row, previous)
        if previous["backend_fingerprint"] != current_embedding_metadata()["backend_fingerprint"]:
            raise ValueError("Select the prior embedding configuration for rollback.")
        collections = repository.open_collections(generation)
        repository.publish_manifest(previous)
        repository.repository_metadata = previous
        repository.collections = collections
        repository.generation = generation
        repository.fingerprint_status = "valid"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline single-writer rebuild into fresh validated collections.")
    parser.add_argument("--path", default="./memory_db", help="Existing PRIMA Chroma store")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Inspect row count and embedding identities")
    mode.add_argument(
        "--rollback", action="store_true", help="Restore a verified prior generation using its configuration"
    )
    args = parser.parse_args(argv)
    if not (Path(args.path) / "chroma.sqlite3").is_file():
        parser.error("existing Chroma store required")
    repo = ChromaMemoryRepository(args.path, inspection=True)
    with repo.transaction():
        notes = repo.list()
        count = len(notes)
        ambiguous = []
        if args.dry_run:
            for note in notes:
                try:
                    get_embedding_pipeline().stored_embedding_source(note)
                except ValueError:
                    ambiguous.append(note.id)
        if args.rollback:
            rollback_embeddings(repo)
            status = "rolled-back"
        elif args.dry_run:
            status = "dry-run"
        else:
            count = rebuild_embeddings(repo)
            status = "rebuilt"
        report: dict[str, Any] = {
            "status": status,
            "count": count,
            "ambiguous_source_ids": ambiguous,
            "requires_rebuild": repo.repository_metadata["backend_fingerprint"] != current_embedding_metadata()["backend_fingerprint"],
            "generation": repo.generation,
            "stored": repo.repository_metadata["backend_fingerprint"],
            "target": current_embedding_metadata(),
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
