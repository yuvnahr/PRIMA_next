from __future__ import annotations

import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

from config.settings import get_settings
from memory.memory_repository import ChromaMemoryRepository
from runtime.prima_runtime import PrimaRuntime


def _ollama_reachable() -> bool:
    settings = get_settings()
    parsed = urlparse(getattr(settings, "ollama_url", "http://localhost:11434"))
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def test_production_embedding_smoke(tmp_path: Path) -> None:
    if not _ollama_reachable():
        pytest.skip("Ollama is not reachable at the configured ollama_url")

    path = tmp_path / "memory_db"
    first = PrimaRuntime(memory_repository=ChromaMemoryRepository(str(path)))
    stored = first.process("Please remember that my name is Nathan. Nathan is called Nate and he likes tea.")
    assert stored.memory_notes_created
    restarted = PrimaRuntime(memory_repository=ChromaMemoryRepository(str(path)))
    result = restarted.answer_question("What does Nate like?")
    assert result.retrieved_memories
    assert restarted.memory_repository.fingerprint_status == "valid"
