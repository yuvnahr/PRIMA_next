"""Select existing PRIMA stores inside a dedicated, session-scoped demo root."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from config.runtime_config import RuntimeConfig
from demo_cli.config import DemoConfig
from memory.embedding_backend import BACKEND_ALIASES, MODEL_BY_BACKEND, EmbeddingBackendConfig
from memory.embedding_pipeline import CanonicalEmbeddingPipeline, use_embedding_pipeline
from memory.experiment_config import EmbeddingExperimentConfig
from memory.memory_repository import ChromaMemoryRepository, InMemoryMemoryRepository, MemoryRepository
from state.state_manager import InMemoryStateManager, JsonStateManager, StateManager

if TYPE_CHECKING:
    from collections.abc import Iterator

    from runtime import PrimaRuntime


class PersistenceError(ValueError):
    """An application-owned message safe to display without backend exception text."""


def session_directory(config: DemoConfig, session_id: str) -> Path:
    try:
        root = config.data_dir.resolve()
        directory = root / "sessions" / hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        # Refuse redirected session paths, including Windows junctions and symlinks.
        for path in (directory, directory / "state.json", directory / "chroma",
                     directory / "chroma" / ".prima_embedding_metadata.json"):
            if path.resolve() != path:
                raise ValueError("Redirected persistence path")
        return directory
    except (OSError, ValueError, RuntimeError):
        raise PersistenceError("Invalid persistent session path. Choose a writable demo data directory without redirected session paths.") from None


@contextmanager
def embedding_scope(config: RuntimeConfig) -> Iterator[None]:
    """Use the same effective embedding configuration as the runtime being constructed."""
    name = BACKEND_ALIASES.get(config.embedding_backend, config.embedding_backend)
    backend = EmbeddingBackendConfig(name, config.embedding_model or MODEL_BY_BACKEND[name], config.embedding_dimensions)
    pipeline = CanonicalEmbeddingPipeline(EmbeddingExperimentConfig(
        backend=name, representation_mode=config.representation_mode, identity_enabled=config.identity_normalization,
    ))
    with use_embedding_pipeline(pipeline, backend):
        yield


def create_stores(
    config: DemoConfig, session_id: str, runtime_config: RuntimeConfig,
) -> tuple[StateManager, MemoryRepository]:
    if not config.persist:
        return InMemoryStateManager(), InMemoryMemoryRepository()
    directory = session_directory(config, session_id)
    repository = None
    try:
        state = JsonStateManager(directory / "state.json")
        with embedding_scope(runtime_config):
            repository = ChromaMemoryRepository(str(directory / "chroma"))
        if repository.fingerprint_status != "valid":
            raise RuntimeError("Embedding fingerprint mismatch")
        return state, repository
    except Exception as exc:
        if repository is not None:
            repository.close()
        if "fingerprint" in str(exc).lower():
            raise PersistenceError("Persistent memory is incompatible with the current embedding configuration. Use the original embedding configuration to reset this session, or choose a new session ID.") from None
        raise PersistenceError("Cannot open persistent demo storage. Check the data directory permissions, state JSON/schema and installed Chroma backend. Existing data was not reset.") from None


def check_storage(config: DemoConfig) -> None:
    """Advisory read-only preflight; strict preflight constructs the actual stores."""
    directory = session_directory(config, config.session_id)
    try:
        import chromadb

        if not callable(chromadb.PersistentClient):
            raise ImportError("Persistent Chroma backend unavailable")
        for path in (directory, directory / "chroma"):
            parent = path
            while not parent.exists():
                parent = parent.parent
            if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
                raise OSError("Demo directory is not writable")
        state_path = directory / "state.json"
        if state_path.exists():
            JsonStateManager(state_path)
    except Exception:
        raise PersistenceError("Persistence is unavailable. Check directory permissions, state JSON/schema and whether the Chroma backend from requirements.txt is installed. Use --strict to verify opening the stores.") from None


def clear_session(config: DemoConfig, session_id: str, runtime: PrimaRuntime) -> None:
    directory = session_directory(config, session_id)
    if (not isinstance(runtime.memory_repository, ChromaMemoryRepository) or
            not isinstance(runtime.state_manager, JsonStateManager) or
            runtime.memory_repository.path.resolve() != directory / "chroma" or
            runtime.state_manager.path.resolve() != directory / "state.json"):
        raise PersistenceError("Persistent reset refused: storage is outside the active session namespace.")
    # No transaction spans JSON and Chroma; leave state untouched until all memory deletion succeeds.
    runtime.memory_index.clear()
    runtime.state_manager.reset(session_id)


def display_data_root(config: DemoConfig) -> str:
    try:
        return os.path.relpath(config.data_dir, Path(__file__).resolve().parents[1])
    except ValueError:  # Windows cannot express a relative path between different drives.
        return str(config.data_dir)
