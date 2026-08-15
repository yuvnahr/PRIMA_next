from __future__ import annotations

import json
import os
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from urllib.parse import urlparse

import pytest

from config.settings import get_settings
from llm.provider import OllamaProvider
from memory.memory_repository import ChromaMemoryRepository
from runtime.prima_runtime import PrimaRuntime


def _ollama_reachable() -> bool:
    settings = get_settings()
    parsed = urlparse(getattr(settings, "ollama_url", "http://localhost:11434"))
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connection_type = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
    connection = connection_type(host, port, timeout=1)
    try:
        connection.request("GET", f"{parsed.path.rstrip('/')}/api/tags")
        response = connection.getresponse()
        if response.status != 200:
            return False
        available = {item["name"] for item in json.loads(response.read()).get("models", [])}
        return os.getenv("PRIMA_LLM_MODEL", OllamaProvider.default_model) in available
    except OSError:
        return False
    finally:
        connection.close()


def test_production_embedding_smoke(tmp_path: Path) -> None:
    if not _ollama_reachable():
        pytest.skip("The configured Ollama endpoint/model is unavailable")

    path = tmp_path / "memory_db"
    first = PrimaRuntime(memory_repository=ChromaMemoryRepository(str(path)))
    stored = first.process("Please remember that my name is Nathan. Nathan is called Nate and he likes tea.")
    assert stored.memory_notes_created
    restarted = PrimaRuntime(memory_repository=ChromaMemoryRepository(str(path)))
    result = restarted.answer_question("What does Nate like?")
    assert result.retrieved_memories
    assert restarted.memory_repository.fingerprint_status == "valid"
