"""Small deterministic production validation for the reasoning controller."""

from __future__ import annotations

import json
from pathlib import Path

from memory.memory_note import MemoryNote
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_confidence import RetrievalConfidence
from memory.retrieval.retrieval_controller import RetrievalResponse
from memory.retrieval.retrieval_result import RetrievalResult
from reasoning.controller import ReasoningController
from reasoning.models import ReasoningMode, ReasoningRequest


def _response(note_id: str, text: str) -> RetrievalResponse:
    note = MemoryNote.create(text, memory_type=MemoryType.SEMANTIC, note_id=note_id, embedding=(0.0,))
    result = RetrievalResult(note=note, score=0.9)
    return RetrievalResponse((result,), RetrievalConfidence(0.9, 0.1, 1.0, 0.9))


def _fixture() -> dict[str, RetrievalResponse]:
    return {"What fact is linked to Alpha?": _response("a", "Alpha identifies Bridge."), "Bridge": _response("b", "Bridge has the linked fact C.")}


def main() -> int:
    calls: list[str] = []
    responses = _fixture()
    result = ReasoningController().answer(
        ReasoningRequest("What fact is linked to Alpha?", mode=ReasoningMode.DIAGNOSTIC),
        retrieve=lambda query: calls.append(query) or responses.get(query, RetrievalResponse((), RetrievalConfidence(0.0, 1.0, 0.0, 0.0))),
        synthesize=lambda _question, evidence: (evidence[-1].note.content, False, (), {"llm_used": False}),
    )
    passed = (
        calls == ["What fact is linked to Alpha?", "Bridge"]
        and result.hop_count == 2
        and result.stop_reason == "sufficient"
    )
    payload = {"passed": passed, "calls": calls, "result": result.to_dict()}
    results_dir = Path("evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    for name in ("reasoning_controller_validation.json", "reasoning_smoke_results.json"):
        (results_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
