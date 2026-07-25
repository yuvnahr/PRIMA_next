"""Focused reasoning/reflection boundary checks."""

from memory.memory_note import MemoryNote
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_confidence import RetrievalConfidence
from memory.retrieval.retrieval_controller import RetrievalResponse
from memory.retrieval.retrieval_result import RetrievalResult
from reasoning.controller import ReasoningController
from reasoning.models import ReasoningBudget, ReasoningMode, ReasoningRequest, SufficiencyStatus
from reasoning.reflection_advisor import ReflectionAction, ReflectionAdvice, ReflectionEvent


def _response(note_id: str, text: str, *, eligible: bool = True) -> RetrievalResponse:
    note = MemoryNote.create(text, MemoryType.SEMANTIC, note_id=note_id, embedding=(0.0,), context={"evidence_eligible": eligible})
    return RetrievalResponse((RetrievalResult(note, 0.9),), RetrievalConfidence(0.9, 0.1, 1.0, 0.9))


def _synthesize(_question, evidence):
    return evidence[-1].note.content, False, (), {}


class _Advisor:
    def __init__(self, query: str) -> None:
        self.query = query
        self.calls = 0

    def advise(self, event, _state, **_kwargs):
        self.calls += 1
        return ReflectionAdvice(True, event, ReflectionAction.BROADEN_QUERY, 0.9, self.query)


def test_one_reflection_intervention_is_bounded_and_applied() -> None:
    advisor = _Advisor("Bridge details")
    responses = {
        "What fact is linked to Alpha?": _response("alpha", "Alpha identifies Bridge."),
        "Bridge": _response("alpha", "Alpha identifies Bridge."),
        "Bridge details": _response("bridge", "Bridge has the linked fact C."),
    }
    result = ReasoningController(reflection_advisor=advisor).answer(
        ReasoningRequest("What fact is linked to Alpha?", mode=ReasoningMode.DIAGNOSTIC, budget=ReasoningBudget(max_hops=3)),
        retrieve=lambda query: responses[query], synthesize=_synthesize,
    )
    assert advisor.calls == 1
    assert result.status is SufficiencyStatus.SUFFICIENT
    assert any(event.event_type == "ReflectionApplied" for event in result.trace_summary)


def test_duplicate_advice_is_rejected() -> None:
    advisor = _Advisor("Bridge")
    result = ReasoningController(reflection_advisor=advisor).answer(
        ReasoningRequest("What fact is linked to Alpha?", mode=ReasoningMode.DIAGNOSTIC),
        retrieve=lambda _query: _response("alpha", "Alpha identifies Bridge."), synthesize=_synthesize,
    )
    assert result.stop_reason == "no_new_evidence"
    assert any(event.event_type == "ReflectionRejected" for event in result.trace_summary)


