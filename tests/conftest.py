"""Test path configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _offline_default_provider(monkeypatch, request):
    from config import settings

    if request.node.name == "test_production_embedding_smoke" and os.getenv("PRIMA_RUN_OLLAMA_SMOKE") == "1":
        return

    monkeypatch.setattr(settings, "_SETTINGS", settings.Settings(
        _env_file=None, default_provider="local", openai_api_key=None, anthropic_api_key=None,
        openai_url="http://127.0.0.1:1", anthropic_url="http://127.0.0.1:1",
        ollama_url="http://127.0.0.1:1", lmstudio_url="http://127.0.0.1:1",
    ))
