"""Focused deterministic checks for the bounded reasoning controller."""

from __future__ import annotations

from memory.memory_note import MemoryNote
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_confidence import RetrievalConfidence
from memory.retrieval.retrieval_controller import RetrievalResponse
from memory.retrieval.retrieval_result import RetrievalResult
from reasoning.controller import ReasoningController
from reasoning.models import ReasoningBudget, ReasoningMode, ReasoningRequest, SufficiencyStatus


def _response(source_id: str, text: str) -> RetrievalResponse:
    result = RetrievalResult(MemoryNote.create(text, MemoryType.SEMANTIC, note_id=source_id, embedding=(0.0,)), 0.9)
    return RetrievalResponse((result,), RetrievalConfidence(0.9, 0.1, 1.0, 0.9))


def _answer(_question: str, evidence: tuple[RetrievalResult, ...]) -> tuple[str, bool, tuple[str, ...], dict[str, object]]:
    return evidence[-1].note.content, False, (), {"llm_used": False}


def test_single_pass_uses_exactly_one_retrieval() -> None:
    calls: list[str] = []
    result = ReasoningController().answer(
        ReasoningRequest("What color is Oak?", mode=ReasoningMode.SINGLE_PASS),
        retrieve=lambda query: calls.append(query) or _response("oak", "Oak color is green."),
        synthesize=_answer,
    )
    assert calls == ["What color is Oak?"]
    assert result.stop_reason == "single_pass"


def test_adaptive_two_hop_acquires_explicit_bridge() -> None:
    calls: list[str] = []
    responses = {
        "What fact is linked to Alpha?": _response("alpha", "Alpha identifies Bridge."),
        "Bridge": _response("bridge", "Bridge has the linked fact C."),
    }
    result = ReasoningController().answer(
        ReasoningRequest("What fact is linked to Alpha?", mode=ReasoningMode.DIAGNOSTIC),
        retrieve=lambda query: calls.append(query) or responses[query],
        synthesize=_answer,
    )
    assert calls == ["What fact is linked to Alpha?", "Bridge"]
    assert result.status is SufficiencyStatus.SUFFICIENT
    assert result.hop_count == 2
    assert [item.source_id for item in result.evidence_references] == ["alpha", "bridge"]


def test_duplicate_or_no_new_evidence_stops() -> None:
    result = ReasoningController().answer(
        ReasoningRequest("What fact is linked to Alpha?", budget=ReasoningBudget(no_progress_limit=1)),
        retrieve=lambda _query: _response("alpha", "Alpha identifies Bridge."),
        synthesize=_answer,
    )
    assert result.stop_reason == "no_new_evidence"


def test_contradictory_evidence_abstains() -> None:
    calls = 0
    def retrieve(_query: str) -> RetrievalResponse:
        nonlocal calls
        calls += 1
        first = _response("one", "River is blue.").results[0]
        second = _response("two", "River is not blue.").results[0]
        return RetrievalResponse((first, second), RetrievalConfidence(0.9, 0.1, 1.0, 0.9))
    result = ReasoningController().answer(ReasoningRequest("What color is River?"), retrieve=retrieve, synthesize=_answer)
    assert calls == 1
    assert result.status is SufficiencyStatus.CONTRADICTORY
    assert result.stop_reason == "contradictory"


def test_trace_has_no_hidden_reasoning_field() -> None:
    result = ReasoningController().answer(
        ReasoningRequest("What color is Oak?", mode=ReasoningMode.DIAGNOSTIC),
        retrieve=lambda _query: _response("oak", "Oak color is green."),
        synthesize=_answer,
    )
    assert all("thought" not in event.to_dict() and "reasoning" not in event.to_dict() for event in result.trace_summary)
