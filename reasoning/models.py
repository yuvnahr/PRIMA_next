"""Typed, request-scoped models for bounded evidence acquisition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any
from uuid import uuid4

from memory.retrieval.retrieval_result import RetrievalResult


class ReasoningMode(str, Enum):
    BYPASS = "bypass"
    SINGLE_PASS = "single_pass"
    ADAPTIVE = "adaptive"
    DELIBERATIVE = "deliberative"
    DIAGNOSTIC = "diagnostic"


class Route(str, Enum):
    BYPASS = "bypass"
    SINGLE_PASS = "single_pass"
    ADAPTIVE = "adaptive"
    CLARIFY = "clarify"


class SufficiencyStatus(str, Enum):
    SUFFICIENT = "sufficient"
    NEED_MORE_EVIDENCE = "need_more_evidence"
    AMBIGUOUS = "ambiguous"
    CONTRADICTORY = "contradictory"
    UNANSWERABLE = "unanswerable"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ReasoningBudget:
    max_hops: int = 3
    max_retrieval_calls: int = 3
    max_llm_calls: int = 4
    max_documents: int = 12
    max_context_tokens: int = 1600
    time_budget_seconds: float = 15.0
    no_progress_limit: int = 1

    max_reflection_interventions: int = 1
    reflection_confidence_threshold: float = 0.6
    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ReasoningRequest:
    question: str
    session_id: str = ""
    mode: ReasoningMode = ReasoningMode.ADAPTIVE
    budget: ReasoningBudget = field(default_factory=ReasoningBudget)
    caller_metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"reasoning_{uuid4()}")

    def __post_init__(self) -> None:
        forbidden = {"ground_truth", "gold_answer", "expected_answer", "gold_supporting_facts", "correct_answer", "is_gold", "truth"}
        if forbidden & {str(key).lower() for key in self.caller_metadata}:
            raise ValueError("Benchmark gold fields are not accepted by production reasoning.")


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    text: str
    source_id: str
    source_type: str
    retrieval_score: float
    hop: int
    query: str
    provenance: dict[str, Any] = field(default_factory=dict)
    result: RetrievalResult | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class InformationNeed:
    need_id: str
    description: str
    query: str
    relation_target: str = ""
    priority: int = 1
    status: str = "open"


@dataclass(frozen=True, slots=True)
class SufficiencyDecision:
    status: SufficiencyStatus
    confidence: float
    answerable: bool
    missing_information: tuple[str, ...] = ()
    unresolved_entities: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    recommended_action: str = "stop"
    reason_code: str = ""


@dataclass(frozen=True, slots=True)
class ReasoningTraceEvent:
    event_type: str
    hop: int
    timestamp: float = field(default_factory=time)
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.event_type, "hop": self.hop, "timestamp": self.timestamp, **self.fields}


@dataclass(slots=True)
class EvidenceState:
    request: ReasoningRequest
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    unresolved_needs: list[InformationNeed] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    attempted_queries: list[str] = field(default_factory=list)
    confidence: float = 0.0
    retrieval_calls: int = 0
    llm_calls: int = 0
    no_progress_hops: int = 0
    started_at: float = field(default_factory=time)
    reflection_interventions: int = 0
    last_reflection_advice: str = ""
    reflection_advice_confidence: float = 0.0
    trace: list[ReasoningTraceEvent] = field(default_factory=list)

    @property
    def context_tokens(self) -> int:
        return sum(max(1, len(item.text.split())) for item in self.evidence_items)

    def add_trace(self, event_type: str, hop: int, **fields: Any) -> None:
        self.trace.append(ReasoningTraceEvent(event_type=event_type, hop=hop, fields=fields))


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    status: SufficiencyStatus
    confidence: float
    evidence_references: tuple[EvidenceItem, ...]
    hop_count: int
    stop_reason: str
    effective_budget: ReasoningBudget
    trace_summary: tuple[ReasoningTraceEvent, ...] = ()
    errors: tuple[str, ...] = ()
    answer_diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def final_response(self) -> str:
        """Compatibility accessor for existing runtime callers."""
        return self.answer

    @property
    def text(self) -> str:
        return self.answer

    @property
    def retrieved_memories(self) -> tuple[RetrievalResult, ...]:
        return tuple(item.result for item in self.evidence_references if item.result is not None)

    def __str__(self) -> str:
        return self.answer

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status.value,
            "confidence": self.confidence,
            "evidence_references": [
                {"source_id": item.source_id, "score": item.retrieval_score, "hop": item.hop, "query": item.query}
                for item in self.evidence_references
            ],
            "hop_count": self.hop_count,
            "stop_reason": self.stop_reason,
            "effective_budget": self.effective_budget.to_dict(),
            "trace_summary": [event.to_dict() for event in self.trace_summary],
            "errors": list(self.errors),
            "answer_diagnostics": dict(self.answer_diagnostics),
        }
