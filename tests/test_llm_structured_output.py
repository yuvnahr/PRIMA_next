from __future__ import annotations

import json

import pytest

from llm.llm_types import LLMRequest, LLMResponse
from llm.provider import OllamaProvider, ProviderError, post_json
from memory.memory_note import MemoryNote
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_result import RetrievalResult
from runtime.prima_runtime import PrimaRuntime


def test_ollama_sends_optional_system_prompt_and_schema(monkeypatch) -> None:
    captured = {}

    def fake_post(url, payload, timeout):
        captured.update({"url": url, "payload": payload, "timeout": timeout})
        return {"response": '{"answer":"filmmaker","evidence":["M1"],"insufficient_information":false}'}

    monkeypatch.setattr("llm.provider.post_json", fake_post)
    schema = {"type": "object"}
    response = OllamaProvider(settings=object()).send(
        LLMRequest(model="qwen3.5:4b", prompt="question", system_prompt="short answers", response_format=schema)
    )

    assert captured["payload"]["system"] == "short answers"
    assert captured["payload"]["format"] == schema
    assert captured["payload"]["think"] is False
    assert response.text.startswith("{")


def test_post_json_rejects_non_http_urls() -> None:
    with pytest.raises(ProviderError, match="http or https"):
        post_json("file:///etc/passwd", {})


def _evidence(note_id: str, text: str, score: float) -> RetrievalResult:
    return RetrievalResult(MemoryNote.create(text, MemoryType.SEMANTIC, note_id=note_id, embedding=(0.0,)), score)


def test_runtime_extracts_answer_and_validated_citations(monkeypatch, tmp_path) -> None:
    raw = json.dumps({"answer": "filmmaker", "evidence": ["M1", "M2", "M1"], "insufficient_information": False})
    captured = {}

    class FakeClient:
        def __init__(self, provider_name): captured["provider"] = provider_name
        def chat(self, **kwargs): captured.update(kwargs); return LLMResponse(raw, raw={"response": raw}, provider="ollama")

    monkeypatch.setattr("runtime.prima_runtime.LLMClient", FakeClient)
    runtime = PrimaRuntime(log_path=tmp_path / "runtime.log")
    answer, used, errors, diagnostics = runtime._synthesize_evidence(
        "Shared profession?",
        (_evidence("sam", "Sam is a filmmaker.", 0.9), _evidence("ruby", "Ruby is a filmmaker.", 0.8)),
        provider="ollama",
        model="qwen3.5:4b",
        max_context_tokens=1600,
    )

    assert answer == "filmmaker"
    assert used is True and errors == ()
    assert diagnostics["selected_memory_ids"] == ["sam", "ruby"]
    assert diagnostics["raw_model_response"] == raw
    assert diagnostics["structured_answer_valid"] is True
    assert captured["system_prompt"].startswith("You are a factual")
    assert captured["response_format"]["required"] == ["answer", "evidence", "insufficient_information"]


def test_runtime_preserves_malformed_raw_response(monkeypatch, tmp_path) -> None:
    class FakeClient:
        def __init__(self, provider_name): pass
        def chat(self, **kwargs): return LLMResponse("verbose unstructured answer")

    monkeypatch.setattr("runtime.prima_runtime.LLMClient", FakeClient)
    runtime = PrimaRuntime(log_path=tmp_path / "runtime.log")
    answer, _, errors, diagnostics = runtime._synthesize_evidence(
        "Question?", (_evidence("one", "Evidence.", 0.9),),
        provider="ollama", model="qwen3.5:4b", max_context_tokens=1600,
    )

    assert answer == "verbose unstructured answer"
    assert errors and diagnostics["structured_answer_valid"] is False
    assert diagnostics["selected_memory_ids"] == []


def test_runtime_recovers_answer_from_truncated_structured_response(monkeypatch, tmp_path) -> None:
    raw = '{"answer":"Stacey Kent","evidence":["M1","M2"],"'

    class FakeClient:
        def __init__(self, provider_name): pass
        def chat(self, **kwargs): return LLMResponse(raw)

    monkeypatch.setattr("runtime.prima_runtime.LLMClient", FakeClient)
    runtime = PrimaRuntime(log_path=tmp_path / "runtime.log")
    answer, _, errors, diagnostics = runtime._synthesize_evidence(
        "Which jazz singer?", (_evidence("one", "Stacey Kent is a jazz singer.", 0.9),),
        provider="ollama", model="qwen3.5:4b", max_context_tokens=1600,
    )

    assert answer == "Stacey Kent"
    assert errors and diagnostics["raw_model_response"] == raw
    assert diagnostics["structured_answer_valid"] is False

def test_runtime_rejects_unknown_context_label(tmp_path) -> None:
    runtime = PrimaRuntime(log_path=tmp_path / "runtime.log")
    from runtime.context_builder import RuntimeContextBuilder
    answer_context = RuntimeContextBuilder().build("Q?", (_evidence("one", "Evidence.", 0.9),))
    with pytest.raises(ValueError, match="unknown context labels"):
        runtime._parse_structured_answer(
            '{"answer":"x","evidence":["M2"],"insufficient_information":false}', answer_context
        )
