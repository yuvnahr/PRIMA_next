"""Embedding backend selection and CLI tests."""

from __future__ import annotations

import pytest

from memory import embedding_backend


class DummySentenceTransformer:
    def __init__(self, model_identifier: str, trust_remote_code: bool = False) -> None:
        self.model_identifier = model_identifier
        self.trust_remote_code = trust_remote_code

    def encode(self, text: str, normalize_embeddings: bool = True) -> list[float]:
        assert normalize_embeddings is True
        if "different" in text or "recipe" in text:
            return [0.0, 1.0, 0.0]
        return [1.0, 0.0, 0.0]


@pytest.fixture(autouse=True)
def reset_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRIMA_EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("PRIMA_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("PRIMA_EMBEDDING_DIMENSIONS", raising=False)
    embedding_backend.reset_embedding_backend_cache()


def test_unknown_backend_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIMA_EMBEDDING_BACKEND", "does-not-exist")

    with pytest.raises(ValueError, match="Unknown PRIMA_EMBEDDING_BACKEND"):
        embedding_backend.embedding_backend_config()


def test_minilm_backend_initializes_selected_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_model(model_identifier: str, trust_remote_code: bool = False) -> DummySentenceTransformer:
        calls.append((model_identifier, trust_remote_code))
        return DummySentenceTransformer(model_identifier, trust_remote_code)

    monkeypatch.setenv("PRIMA_EMBEDDING_BACKEND", "minilm")
    monkeypatch.setattr(embedding_backend, "_sentence_transformer_model", fake_model)

    backend = embedding_backend.get_embedding_backend()

    assert backend.name == "minilm"
    assert backend.model_identifier == "sentence-transformers/all-MiniLM-L6-v2"
    assert calls == [("sentence-transformers/all-MiniLM-L6-v2", False)]


def test_bge_small_backend_initializes_selected_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_model(model_identifier: str, trust_remote_code: bool = False) -> DummySentenceTransformer:
        calls.append((model_identifier, trust_remote_code))
        return DummySentenceTransformer(model_identifier, trust_remote_code)

    monkeypatch.setenv("PRIMA_EMBEDDING_BACKEND", "bge-small")
    monkeypatch.setattr(embedding_backend, "_sentence_transformer_model", fake_model)

    backend = embedding_backend.get_embedding_backend()

    assert backend.name == "bge_small"
    assert backend.model_identifier == "BAAI/bge-small-en-v1.5"
    assert calls == [("BAAI/bge-small-en-v1.5", False)]


def test_nomic_backend_initializes_with_trust_remote_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_model(model_identifier: str, trust_remote_code: bool = False) -> DummySentenceTransformer:
        calls.append((model_identifier, trust_remote_code))
        return DummySentenceTransformer(model_identifier, trust_remote_code)

    monkeypatch.setenv("PRIMA_EMBEDDING_BACKEND", "nomic")
    monkeypatch.setattr(embedding_backend, "_sentence_transformer_model", fake_model)

    backend = embedding_backend.get_embedding_backend()

    assert backend.name == "nomic"
    assert backend.model_identifier == "nomic-ai/nomic-embed-text-v1.5"
    assert calls == [("nomic-ai/nomic-embed-text-v1.5", True)]


def test_embed_text_uses_selected_provider_without_stable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_model(model_identifier: str, trust_remote_code: bool = False) -> DummySentenceTransformer:
        return DummySentenceTransformer(model_identifier, trust_remote_code)

    monkeypatch.setenv("PRIMA_EMBEDDING_BACKEND", "minilm")
    monkeypatch.setattr(embedding_backend, "_sentence_transformer_model", fake_model)

    vector = embedding_backend.embed_text("selected learned provider", dimensions=3)

    assert vector == [1.0, 0.0, 0.0]
    assert vector != embedding_backend.stable_hash_embedding("selected learned provider", dimensions=3)


def test_self_test_reports_pass_for_selected_learned_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_model(model_identifier: str, trust_remote_code: bool = False) -> DummySentenceTransformer:
        return DummySentenceTransformer(model_identifier, trust_remote_code)

    monkeypatch.setenv("PRIMA_EMBEDDING_BACKEND", "bge_small")
    monkeypatch.setattr(embedding_backend, "_sentence_transformer_model", fake_model)

    exit_code = embedding_backend.main(["--self-test"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "active backend: bge_small" in output
    assert "embedding dimension: 64" in output
    assert "model loaded: yes" in output
    assert "pass/fail: pass" in output
