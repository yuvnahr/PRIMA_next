from __future__ import annotations

import json
import math
import os
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from urllib.parse import urlparse

import pytest

from config.settings import get_settings
from llm.provider import OllamaProvider
from memory.embedding_backend import (
    EmbeddingBackendConfig,
    get_embedding_backend,
    use_embedding_backend_config,
    validate_learned_backend,
)
from memory.memory_repository import ChromaMemoryRepository
from runtime.prima_runtime import PrimaRuntime


@pytest.mark.skipif(os.getenv("PRIMA_RUN_LEARNED_EMBEDDING_SMOKE") != "1", reason="Opt-in pinned-cache CPU model check")
def test_pinned_cpu_embedding_semantics() -> None:
    config = EmbeddingBackendConfig("minilm", "sentence-transformers/all-MiniLM-L6-v2", 384,
                                    revision="c9745ed1d9f207416be6d2e6f8de32d1f16199bf", device="cpu")
    with use_embedding_backend_config(config):
        result = validate_learned_backend()
        vector = get_embedding_backend().embed("A separate finite normalized vector check.")
    assert result["passed"] and result["backend"]["device"] == "cpu"
    assert result["backend"]["embedding_dimensions"] == 384
    assert result["backend"]["revision"] == config.revision
    assert result["backend"]["snapshot"]["artifact_sha256"] == "7550e754abbd9086d0f997f2fda4ff22744ac6815bea4550123b73e40b528a7e"
    assert len(vector) == 384 and all(math.isfinite(value) for value in vector)
    assert math.isclose(sum(value * value for value in vector), 1.0, abs_tol=1e-5)


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
