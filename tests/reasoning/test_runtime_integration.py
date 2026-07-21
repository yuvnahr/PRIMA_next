"""Public runtime smoke coverage for adaptive evidence acquisition."""

from __future__ import annotations

from memory.memory_note import MemoryNote
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_confidence import RetrievalConfidence
from memory.retrieval.retrieval_controller import RetrievalResponse
from memory.retrieval.retrieval_result import RetrievalResult
from runtime import PrimaRuntime


def _response(source_id: str, text: str) -> RetrievalResponse:
    result = RetrievalResult(MemoryNote.create(text, MemoryType.SEMANTIC, note_id=source_id, embedding=(0.0,)), 0.9)
    return RetrievalResponse((result,), RetrievalConfidence(0.9, 0.1, 1.0, 0.9))


def test_runtime_uses_production_retrieval_for_every_adaptive_hop(tmp_path) -> None:
    responses = {
        "What fact is linked to Alpha?": _response("alpha", "Alpha identifies Bridge."),
        "Bridge": _response("bridge", "Bridge has the linked fact C."),
    }

    class FakeRetrievalController:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def retrieve(self, request):
            self.calls.append(request.query)
            return responses[request.query]

    runtime = PrimaRuntime(log_path=tmp_path / "runtime.log")
    fake = FakeRetrievalController()
    runtime.retrieval_controller = fake
    runtime._synthesize_evidence = lambda _q, evidence, **_kwargs: (evidence[-1].note.content, False, (), {"llm_used": False})
    result = runtime.answer_question("What fact is linked to Alpha?", reasoning_mode="adaptive", diagnostics=True)

    assert fake.calls == ["What fact is linked to Alpha?", "Bridge"]
    assert result.hop_count == 2
    assert result.stop_reason == "sufficient"
    assert [item.source_id for item in result.evidence_references] == ["alpha", "bridge"]
