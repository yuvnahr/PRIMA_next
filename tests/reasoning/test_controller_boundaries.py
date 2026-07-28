"""Budget, failure, and isolation checks for bounded reasoning."""

from __future__ import annotations

from time import sleep

from memory.memory_note import MemoryNote
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_confidence import RetrievalConfidence
from memory.retrieval.retrieval_controller import RetrievalResponse
from memory.retrieval.retrieval_result import RetrievalResult
from reasoning.controller import ReasoningController
from reasoning.models import (
    EvidenceState,
    ReasoningBudget,
    ReasoningMode,
    ReasoningRequest,
    SufficiencyStatus,
)
from reasoning.stopping_policy import StoppingPolicy


def _response(source_id: str, text: str) -> RetrievalResponse:
    result = RetrievalResult(MemoryNote.create(text, MemoryType.SEMANTIC, note_id=source_id, embedding=(0.0,)), 0.9)
    return RetrievalResponse((result,), RetrievalConfidence(0.9, 0.1, 1.0, 0.9))


def _answer(_question, evidence):
    return evidence[-1].note.content, False, (), {"llm_used": False}


def test_bypass_and_empty_question_do_not_retrieve() -> None:
    calls: list[str] = []
    controller = ReasoningController()
    bypass = controller.answer(ReasoningRequest("hello", mode=ReasoningMode.BYPASS), retrieve=lambda q: calls.append(q), synthesize=lambda q, e: ("hello", False, (), {}))
    clarify = controller.answer(ReasoningRequest(""), retrieve=lambda q: calls.append(q), synthesize=_answer)
    assert calls == []
    assert bypass.stop_reason == "bypass"
    assert clarify.status is SufficiencyStatus.AMBIGUOUS


def test_first_hop_sufficient_never_requests_a_follow_up() -> None:
    calls: list[str] = []
    result = ReasoningController().answer(
        ReasoningRequest("What color is Oak?"),
        retrieve=lambda q: calls.append(q) or _response("oak", "Oak color is green."),
        synthesize=_answer,
    )
    assert calls == ["What color is Oak?"]
    assert result.stop_reason == "sufficient"


def test_hop_and_retrieval_budgets_are_enforced() -> None:
    responses = {
        "What final trait has Alpha?": _response("a", "Alpha identifies Bridge."),
        "Bridge": _response("b", "Bridge identifies Delta."),
        "Delta": _response("c", "Delta identifies Echo."),
    }
    max_hops = ReasoningController().answer(
        ReasoningRequest("What final trait has Alpha?", budget=ReasoningBudget(max_hops=3, max_retrieval_calls=5, no_progress_limit=9)),
        retrieve=lambda q: responses[q], synthesize=_answer,
    )
    max_calls = ReasoningController().answer(
        ReasoningRequest("What final trait has Alpha?", budget=ReasoningBudget(max_retrieval_calls=1, no_progress_limit=9)),
        retrieve=lambda q: responses[q], synthesize=_answer,
    )
    assert max_hops.stop_reason == "max_hops"
    assert max_calls.stop_reason == "max_retrieval_calls"


def test_time_and_context_budgets_abstain() -> None:
    slow = ReasoningController().answer(
        ReasoningRequest("What final trait has Alpha?", budget=ReasoningBudget(time_budget_seconds=0.001)),
        retrieve=lambda _q: (sleep(0.01) or _response("a", "Alpha identifies Bridge.")), synthesize=_answer,
    )
    large_text = "Alpha identifies Bridge. " + "filler " * 130
    bounded = ReasoningController().answer(
        ReasoningRequest("What final trait has Alpha?", budget=ReasoningBudget(max_context_tokens=128)),
        retrieve=lambda _q: _response("a", large_text), synthesize=_answer,
    )
    assert slow.stop_reason == "time_budget"
    assert bounded.stop_reason == "context_budget"


def test_retrieval_failure_and_no_evidence_are_structured() -> None:
    failure = ReasoningController().answer(ReasoningRequest("What color is Oak?"), retrieve=lambda _q: (_ for _ in ()).throw(RuntimeError("offline")), synthesize=_answer)
    empty = ReasoningController().answer(
        ReasoningRequest("What color is Oak?"),
        retrieve=lambda _q: RetrievalResponse((), RetrievalConfidence(0.0, 1.0, 0.0, 0.0)), synthesize=_answer,
    )
    assert failure.status is SufficiencyStatus.ERROR
    assert failure.stop_reason == "error"
    assert empty.status is SufficiencyStatus.UNANSWERABLE


def test_duplicate_query_policy_and_request_isolation() -> None:
    state = EvidenceState(ReasoningRequest("Question"))
    state.attempted_queries.append("Bridge")
    assert StoppingPolicy().stop_reason(state, __import__("reasoning.models", fromlist=["SufficiencyDecision"]).SufficiencyDecision(SufficiencyStatus.NEED_MORE_EVIDENCE, 0.0, False), next_query="bridge") == "duplicate_query"
    first = ReasoningController().answer(ReasoningRequest("What color is Oak?", session_id="one"), retrieve=lambda _q: _response("oak", "Oak color is green."), synthesize=_answer)
    second = ReasoningController().answer(ReasoningRequest("What color is Pine?", session_id="two"), retrieve=lambda _q: _response("pine", "Pine color is blue."), synthesize=_answer)
    assert [item.source_id for item in first.evidence_references] == ["oak"]
    assert [item.source_id for item in second.evidence_references] == ["pine"]
