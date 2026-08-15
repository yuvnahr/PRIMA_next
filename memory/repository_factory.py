"""Memory repository selection and fail-fast mode validation."""

from __future__ import annotations

from dataclasses import dataclass

from config.runtime_mode import RuntimeMode
from memory.memory_repository import ChromaMemoryRepository, InMemoryMemoryRepository, MemoryRepository


@dataclass(frozen=True, slots=True)
class RepositorySelection:
    """Visible memory repository identity for diagnostics and manifests."""

    backend: str
    persistent: bool
    path: str | None = None


def select_memory_repository(
    mode: RuntimeMode,
    repository: MemoryRepository | None = None,
    *,
    backend: str | None = None,
    path: str | None = None,
) -> tuple[MemoryRepository, RepositorySelection]:
    """Select a repository without silent production or benchmark defaults."""

    if repository is not None:
        selected = _describe(repository)
        if mode is RuntimeMode.PRODUCTION and not selected.persistent:
            raise RuntimeError("Production mode requires configured persistent ChromaDB memory storage.")
        if mode is RuntimeMode.BENCHMARK and backend is None:
            raise RuntimeError("Benchmark mode must declare its memory repository choice in the manifest.")
        if backend is not None and backend != selected.backend:
            raise ValueError(
                f"Configured memory backend {backend!r} does not match injected {selected.backend!r} repository."
            )
        return repository, selected
    if mode is RuntimeMode.TEST and backend in {None, "in_memory"}:
        return InMemoryMemoryRepository(), RepositorySelection("in_memory", False)
    if mode is RuntimeMode.BENCHMARK and backend is None:
        raise RuntimeError("Benchmark mode must declare its memory repository choice in the manifest.")
    if mode is RuntimeMode.PRODUCTION and (backend != "chroma" or not path):
        raise RuntimeError("Production preflight requires memory_backend='chroma' and a configured memory_path.")
    if backend == "in_memory":
        if mode is RuntimeMode.PRODUCTION:
            raise RuntimeError("Production mode cannot use ephemeral in-memory storage.")
        return InMemoryMemoryRepository(), RepositorySelection("in_memory", False)
    if backend == "chroma" and path:
        repository = ChromaMemoryRepository(path)
        if repository.fingerprint_status != "valid":
            raise RuntimeError(f"ChromaDB preflight failed: fingerprint status is {repository.fingerprint_status!r}.")
        return repository, RepositorySelection("chroma", True, path)
    raise ValueError(f"Unsupported memory repository configuration: backend={backend!r}, path={path!r}.")


def _describe(repository: MemoryRepository) -> RepositorySelection:
    if isinstance(repository, ChromaMemoryRepository):
        if repository.fingerprint_status != "valid":
            raise RuntimeError(f"ChromaDB preflight failed: fingerprint status is {repository.fingerprint_status!r}.")
        return RepositorySelection("chroma", True, str(repository.path))
    if isinstance(repository, InMemoryMemoryRepository):
        return RepositorySelection("in_memory", False)
    raise TypeError(f"Unsupported memory repository type: {type(repository).__name__}.")
