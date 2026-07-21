"""The canonical bounded evidence-acquisition controller."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from memory.retrieval.retrieval_controller import RetrievalResponse
from reasoning.evidence_integrator import EvidenceIntegrator
from reasoning.information_need_generator import InformationNeedGenerator
from reasoning.models import AnswerResult, EvidenceState, ReasoningRequest, Route, SufficiencyStatus
from reasoning.stopping_policy import StoppingPolicy
from reasoning.sufficiency_verifier import SufficiencyVerifier
from reasoning.task_analyzer import TaskAnalyzer

Retriever = Callable[[str], RetrievalResponse]
Synthesizer = Callable[[str, tuple[Any, ...]], tuple[str, bool, tuple[str, ...], dict[str, Any]]]


@dataclass(slots=True)
class ReasoningController:
    """Orchestrate retrieval calls without owning retrieval or persistent memory."""

    task_analyzer: TaskAnalyzer = field(default_factory=TaskAnalyzer)
    evidence_integrator: EvidenceIntegrator = field(default_factory=EvidenceIntegrator)
    verifier: SufficiencyVerifier = field(default_factory=SufficiencyVerifier)
    need_generator: InformationNeedGenerator = field(default_factory=InformationNeedGenerator)
    stopping_policy: StoppingPolicy = field(default_factory=StoppingPolicy)

    def answer(self, request: ReasoningRequest, *, retrieve: Retriever, synthesize: Synthesizer) -> AnswerResult:
        state = EvidenceState(request=request)
        route = self.task_analyzer.route(request)
        state.add_trace("ReasoningStarted", 0, mode=request.mode.value, session_id=request.session_id)
        state.add_trace("TaskRouted", 0, route=route.value)
        if route is Route.CLARIFY:
            return self._result(state, "Please clarify the question.", SufficiencyStatus.AMBIGUOUS, "ambiguity")
        if route is Route.BYPASS:
            answer, llm_used, errors, diagnostics = synthesize(request.question, ())
            state.llm_calls += int(llm_used)
            return self._result(state, answer, SufficiencyStatus.SUFFICIENT if answer else SufficiencyStatus.UNANSWERABLE,
                                "bypass", errors, diagnostics)

        query = request.question
        hop = 0
        while True:
            try:
                state.add_trace("RetrievalStarted", hop, query=query)
                response = retrieve(query)
                state.retrieval_calls += 1
            except Exception as exc:
                state.add_trace("ReasoningFailed", hop, error_category=type(exc).__name__)
                return self._result(state, "", SufficiencyStatus.ERROR, "error", (str(exc),))
            integration = self.evidence_integrator.integrate(state, response, hop=hop, query=query)
            state.attempted_queries.append(query)
            state.add_trace(
                "EvidenceRetrieved", hop, source_ids=[item.note.id for item in response.results],
                count=len(response.results), retrieval_confidence=round(float(response.confidence.confidence), 6),
            )
            state.add_trace("EvidenceIntegrated", hop, evidence_count=len(state.evidence_items), novelty_count=integration.added_count)
            decision = self.verifier.evaluate(state)
            state.add_trace("SufficiencyEvaluated", hop, status=decision.status.value, reason_code=decision.reason_code)
            if route is Route.SINGLE_PASS:
                return self._synthesize(state, decision, synthesize, "single_pass")
            need = self.need_generator.generate(state, decision) if decision.status is SufficiencyStatus.NEED_MORE_EVIDENCE else None
            stop_reason = self.stopping_policy.stop_reason(state, decision, next_query=need.query if need else None)
            if stop_reason:
                return self._synthesize(state, decision, synthesize, stop_reason)
            if need is None:
                return self._synthesize(state, decision, synthesize, "unanswerable")
            state.unresolved_needs = [need]
            state.add_trace("InformationNeedCreated", hop + 1, need_id=need.need_id, query=need.query)
            state.add_trace("ReasoningContinued", hop + 1, query=need.query)
            query = need.query
            hop += 1

    def _synthesize(self, state: EvidenceState, decision: Any, synthesize: Synthesizer, stop_reason: str) -> AnswerResult:
        answer = ""
        llm_used = False
        errors: tuple[str, ...] = ()
        diagnostics: dict[str, Any] = {}
        if decision.answerable:
            answer, llm_used, errors, diagnostics = synthesize(
                state.request.question, tuple(item.result for item in state.evidence_items if item.result is not None)
            )
            state.llm_calls += int(llm_used)
            if state.llm_calls > state.request.budget.max_llm_calls:
                return self._result(state, "", SufficiencyStatus.ERROR, "max_llm_calls", ("LLM call budget exceeded.",))
            state.add_trace("AnswerSynthesized", len(state.attempted_queries), llm_used=llm_used)
        elif not answer:
            answer = self._abstention(state.request.question, stop_reason)
        return self._result(state, answer, decision.status, stop_reason, errors, diagnostics)

    def _result(
        self, state: EvidenceState, answer: str, status: SufficiencyStatus, stop_reason: str,
        errors: tuple[str, ...] = (), diagnostics: dict[str, Any] | None = None,
    ) -> AnswerResult:
        state.add_trace("ReasoningStopped", len(state.attempted_queries), stop_reason=stop_reason,
                        hop_count=len(state.attempted_queries), budget_usage={
                            "retrieval_calls": state.retrieval_calls, "llm_calls": state.llm_calls, "context_tokens": state.context_tokens,
                        })
        return AnswerResult(
            answer=answer, status=status, confidence=round(state.confidence, 6), evidence_references=tuple(state.evidence_items),
            hop_count=len(state.attempted_queries), stop_reason=stop_reason, effective_budget=state.request.budget,
            trace_summary=tuple(state.trace) if state.request.mode.value == "diagnostic" else tuple(state.trace[-1:]),
            errors=errors, answer_diagnostics=diagnostics or {},
        )

    def _abstention(self, question: str, reason: str) -> str:
        return f"I do not have sufficient evidence to answer this question ({reason})."
