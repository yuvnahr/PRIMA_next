"""Deterministic checks for the reasoning/reflection boundary."""

from __future__ import annotations

import json
from pathlib import Path

from memory.memory_note import MemoryNote
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_confidence import RetrievalConfidence
from memory.retrieval.retrieval_controller import RetrievalResponse
from memory.retrieval.retrieval_result import RetrievalResult
from reasoning.controller import ReasoningController
from reasoning.models import ReasoningBudget, ReasoningMode, ReasoningRequest, SufficiencyStatus
from reasoning.reflection_advisor import ReflectionAction, ReflectionAdvice


def _response(note_id: str, text: str, eligible: bool = True) -> RetrievalResponse:
    note = MemoryNote.create(text, MemoryType.SEMANTIC, note_id=note_id, embedding=(0.0,), context={"evidence_eligible": eligible})
    return RetrievalResponse((RetrievalResult(note, 0.9),), RetrievalConfidence(0.9, 0.1, 1.0, 0.9))


class _Advisor:
    def advise(self, event, _state, **_kwargs):
        return ReflectionAdvice(True, event, ReflectionAction.BROADEN_QUERY, 0.9, "Bridge details")


def main() -> int:
    responses = {
        "What fact is linked to Alpha?": _response("alpha", "Alpha identifies Bridge."),
        "Bridge": _response("alpha", "Alpha identifies Bridge."),
        "Bridge details": _response("bridge", "Bridge has the linked fact C."),
    }
    result = ReasoningController(reflection_advisor=_Advisor()).answer(
        ReasoningRequest("What fact is linked to Alpha?", mode=ReasoningMode.DIAGNOSTIC, budget=ReasoningBudget(max_hops=3)),
        retrieve=lambda query: responses[query], synthesize=lambda _q, evidence: (evidence[-1].note.content, False, (), {}),
    )
    try:
        ReasoningRequest("x", caller_metadata={"gold_answer": "x"})
    except ValueError:
        gold_isolated = True
    else:
        gold_isolated = False
    excluded = ReasoningController().answer(ReasoningRequest("Oak?"), retrieve=lambda _q: _response("rule", "Broaden.", False), synthesize=lambda *_: ("", False, (), {}))
    payload = {
        "passed": result.status is SufficiencyStatus.SUFFICIENT and gold_isolated and not excluded.evidence_references,
        "reflection_interventions": 1,
        "trace_events": [event.event_type for event in result.trace_summary],
        "procedural_memory_excluded": not excluded.evidence_references,
        "gold_answer_isolated": gold_isolated,
    }
    path = Path("evaluation/results/reasoning_reflection_integration_validation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("PASS" if payload["passed"] else "FAIL")
    return 0 if payload["passed"] else 1

