"""The canonical bounded evidence-acquisition controller."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from llm.request_budget import RequestBudgetExceeded, prepaid_call
from memory.retrieval.retrieval_controller import RetrievalResponse
from reasoning.evidence_audit import evidence_ids, retrieval_hop, summarize_hops
from reasoning.evidence_integrator import EvidenceIntegrator
from reasoning.information_need_generator import InformationNeedGenerator, normalize_query
from reasoning.models import (
    AnswerResult,
    EvidenceState,
    ReasoningRequest,
    Route,
    SufficiencyDecision,
    SufficiencyStatus,
)
from reasoning.reflection_advisor import ReflectionAction, ReflectionAdvisor, ReflectionEvent
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

    reflection_advisor: ReflectionAdvisor | None = None
    def answer(
        self, request: ReasoningRequest, *, retrieve: Retriever, synthesize: Synthesizer | None = None,
        retrieval_hops: list[dict[str, Any]] | None = None, attempt_index: int = 0,
    ) -> AnswerResult:
        state = EvidenceState(request=request, retrieval_hops=retrieval_hops if retrieval_hops is not None else [])
        route = self.task_analyzer.route(request)
        state.add_trace("ReasoningStarted", 0, mode=request.mode.value, session_id=request.session_id)
        state.add_trace("TaskRouted", 0, route=route.value)
        if route is Route.CLARIFY:
            return self._result(state, "Please clarify the question.", SufficiencyStatus.AMBIGUOUS, "ambiguity")
        if route is Route.BYPASS:
            return self._synthesize(state, SufficiencyDecision(SufficiencyStatus.SUFFICIENT, 1.0, True),
                                    synthesize, "bypass", allow_empty=True)

        query = str(request.caller_metadata.get("initial_query", request.question))
        hop = 0
        while True:
            if state.shared_budget.remaining_seconds <= 0:
                return self._synthesize(state, self.verifier.evaluate(state), synthesize, "time_budget")
            if state.shared_budget.retrieval_calls >= request.budget.max_retrieval_calls:
                return self._synthesize(state, self.verifier.evaluate(state), synthesize, "max_retrieval_calls")
            if state.shared_budget.retrieval_time_exhausted:
                return self._synthesize(state, self.verifier.evaluate(state), synthesize, "answer_time_reserve")
            try:
                state.add_trace("RetrievalStarted", hop, query=query)
                started = perf_counter()
                state.shared_budget.retrieval_calls += 1
                response = retrieve(query)
                state.retrieval_calls += 1
            except Exception as exc:
                state.add_trace("ReasoningFailed", hop, error_category=type(exc).__name__)
                return self._result(state, "", SufficiencyStatus.ERROR, "error", (str(exc),))
            audit = retrieval_hop(response, conversation_id=request.session_id, query=query, hop=hop,
                                  elapsed_ms=round((perf_counter() - started) * 1000, 3))
            audit.update(hop_index=len(state.retrieval_hops), attempt_index=attempt_index)
            if attempt_index:
                audit["attempt_kind"] = "workflow_retry"
            state.retrieval_hops.append(audit)
            before = len(state.evidence_items)
            try:
                integration = self.evidence_integrator.integrate(state, response, hop=hop, query=query)
            except Exception:
                audit["stop_reason"] = "integration_error"
                raise
            added = evidence_ids(state.evidence_items[before:], request.session_id)
            audit["added_source_turn_ids"] = added["source_turn_ids"]
            audit["added_memory_ids"] = added["memory_ids"]
            audit["untraceable_added_memory_count"] = added["untraceable_memory_count"]
            audit["untraceable_added_memory_ids"] = sorted({item.source_id for item in state.evidence_items[before:] if not item.provenance.get("source_turn_id")})
            audit["excluded_memory_ids"] = sorted({item.note.id for item in response.results if not item.note.context.get("evidence_eligible", True)})
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
                reflected_query = self._advised_query(state, query, hop, stop_reason, decision)
                if reflected_query:
                    query = reflected_query
                    hop += 1
                    continue
                return self._synthesize(state, decision, synthesize, stop_reason)
            if need is None:
                return self._synthesize(state, decision, synthesize, "no_supported_expansion")
            state.unresolved_needs = [need]
            state.add_trace("InformationNeedCreated", hop + 1, need_id=need.need_id, query=need.query)
            state.add_trace("ReasoningContinued", hop + 1, query=need.query)
            query = need.query
            hop += 1
    def _advised_query(self, state: EvidenceState, query: str, hop: int, stop_reason: str, decision: SufficiencyDecision) -> str | None:
        from uncertainty.execution_gate import UncertaintyGate

        event = {"no_new_evidence": ReflectionEvent.NO_NEW_EVIDENCE, "duplicate_query": ReflectionEvent.DUPLICATE_QUERY, "contradictory": ReflectionEvent.CONTRADICTORY_EVIDENCE}.get(stop_reason)
        budget = state.request.budget
        suppression = UncertaintyGate.reflection_suppression(enabled=self.reflection_advisor is not None,
            remaining_seconds=state.shared_budget.remaining_seconds,
            remaining_calls=state.shared_budget.max_llm_calls - state.shared_budget.llm_calls,
            interventions=state.reflection_interventions, max_interventions=budget.max_reflection_interventions,
            work_available=(not state.shared_budget.retrieval_time_exhausted
                and state.shared_budget.retrieval_calls < budget.max_retrieval_calls and len(state.attempted_queries) < budget.max_hops))
        if event is None or suppression or self.reflection_advisor is None:
            if event is not None:
                state.add_trace(
                    "ReflectionSuppressed",
                    hop,
                    reason_code=suppression or "not_eligible",
                    before_query=query,
                    after_query=query,
                    before_plan=None,
                    after_plan=None,
                    before_answer="",
                    after_answer="",
                    confidence_delta=0.0,
                    utility=0.0,
                )
            return None
        state.add_trace("ReflectionRequested", hop, event=event.value)
        state.reflection_interventions += 1
        try:
            advice = self.reflection_advisor.advise(event, state, query=query, stop_reason=stop_reason)
        except Exception as exc:
            state.add_trace(
                "ReflectionRejected",
                hop,
                reason_code="advisor_error",
                error_category=type(exc).__name__,
                before_query=query,
                after_query=query,
                before_plan=None,
                after_plan=None,
                before_answer="",
                after_answer="",
                confidence_delta=0.0,
                utility=0.0,
            )
            return None
        state.last_reflection_advice = advice.action.value
        state.reflection_advice_confidence = advice.confidence
        state.add_trace("ReflectionGenerated", hop, action=advice.action.value, confidence=round(advice.confidence, 6))
        candidate = self.need_generator.supported_query(state, decision, advice.suggested_query) or ""
        rejected = (
            not advice.trigger
            or advice.action
            not in {
                ReflectionAction.REVISE_QUERY,
                ReflectionAction.BROADEN_QUERY,
                ReflectionAction.PIVOT_ENTITY,
                ReflectionAction.RETRY_TRANSIENT_FAILURE,
            }
            or not candidate
            or advice.confidence < budget.reflection_confidence_threshold
            or advice.contains_evaluation_leakage()
        )
        attempted = {normalize_query(item) for item in state.attempted_queries}
        exceeds_budget = (state.shared_budget.retrieval_calls + 1 > budget.max_retrieval_calls
                          or len(state.attempted_queries) + 1 > budget.max_hops or state.shared_budget.retrieval_time_exhausted)
        rejection_reason = (
            "invalid_advice"
            if rejected
            else "duplicate_advice"
            if normalize_query(candidate) in attempted
            else "budget_exhausted"
            if exceeds_budget
            else ""
        )
        if rejection_reason:
            state.add_trace(
                "ReflectionRejected",
                hop,
                reason_code=rejection_reason,
                before_query=query,
                after_query=candidate or query,
                before_plan=None,
                after_plan=None,
                before_answer="",
                after_answer="",
                confidence_delta=0.0,
                utility=0.0,
            )
            return None
        state.add_trace(
            "ReflectionAccepted",
            hop,
            action=advice.action.value,
            before_query=query,
            after_query=candidate,
            before_plan=None,
            after_plan=None,
            before_answer="",
            after_answer="",
            decision_reason="accepted",
            confidence_delta=round(advice.confidence - state.confidence, 6),
            utility=round(max(0.0, advice.confidence - state.confidence), 6),
        )
        state.add_trace("ReflectionApplied", hop + 1, action=advice.action.value, query=candidate)
        return candidate


    def _synthesize(
        self, state: EvidenceState, decision: SufficiencyDecision, synthesize: Synthesizer | None,
        stop_reason: str, *, allow_empty: bool = False,
    ) -> AnswerResult:
        eligible = [item for item in state.evidence_items if item.text.strip() and item.provenance.get("evidence_eligible", True)]
        blocked = (decision.status in {SufficiencyStatus.ERROR, SufficiencyStatus.CONTRADICTORY,
                                      SufficiencyStatus.AMBIGUOUS, SufficiencyStatus.UNANSWERABLE}
                   or bool(decision.contradictions or state.contradictions))
        bounded = not decision.answerable
        # Canonical acquisition applies the final context gate after workflow compression.
        permission = (not blocked and (bool(eligible) or allow_empty)
                      and (synthesize is None or state.context_tokens <= state.request.budget.max_context_tokens)
                      and not state.shared_budget.denial_reason()
                      and not (bounded and state.shared_budget.bounded_answer_attempts))
        answer = self._abstention(state.request.question, stop_reason)
        errors: tuple[str, ...] = ()
        diagnostics: dict[str, Any] = {}
        outcome = "deferred" if permission and synthesize is None else "blocked"
        if permission and synthesize is not None:
            try:
                state.shared_budget.reserve_call()
                if bounded:
                    state.shared_budget.bounded_answer_attempts += 1
                with prepaid_call(state.shared_budget):
                    answer, llm_used, errors, diagnostics = synthesize(
                        state.request.question, tuple(item.result for item in eligible if item.result is not None))
                if not llm_used:
                    state.shared_budget.llm_calls -= 1
                    if bounded:
                        state.shared_budget.bounded_answer_attempts -= 1
                outcome = "failed" if errors else str(diagnostics.get("outcome", "answered" if answer else "abstained"))
                if state.shared_budget.remaining_seconds <= 0:
                    answer, outcome = self._abstention(state.request.question, "time_budget"), "time_budget"
                state.add_trace("AnswerSynthesized", len(state.attempted_queries), llm_used=llm_used)
            except (RequestBudgetExceeded, RuntimeError, TypeError, ValueError) as exc:
                errors, outcome = (str(exc),), "failed"
        diagnostics.update(answer_attempt={"permitted": permission, "outcome": outcome,
                                           "bounded": bounded, "retrieval_stop_reason": stop_reason,
                                           "denial_reason": state.shared_budget.denial_reason() if not permission else ""})
        return self._result(state, answer, decision.status, stop_reason, errors, diagnostics, permitted=permission)

    def _result(
        self, state: EvidenceState, answer: str, status: SufficiencyStatus, stop_reason: str,
        errors: tuple[str, ...] = (), diagnostics: dict[str, Any] | None = None,
        *, permitted: bool = False,
    ) -> AnswerResult:
        state.llm_calls = state.shared_budget.llm_calls
        state.add_trace("ReasoningStopped", len(state.attempted_queries), stop_reason=stop_reason,
                        hop_count=len(state.attempted_queries), budget_usage={
                            "retrieval_calls": state.retrieval_calls, "llm_calls": state.llm_calls,
                            "context_tokens": state.context_tokens,
                            "reflection_interventions": state.reflection_interventions,
                        })
        result_diagnostics = dict(diagnostics or {})
        result_diagnostics["request_budget"] = state.shared_budget.to_dict()
        if state.retrieval_calls:
            state.retrieval_hops[-1]["stop_reason"] = stop_reason
        result_diagnostics["evidence_audit"] = summarize_hops(state.retrieval_hops)
        result_diagnostics["reflection_attempts"] = [
            event.to_dict()
            for event in state.trace
            if event.event_type in {"ReflectionAccepted", "ReflectionRejected", "ReflectionSuppressed"}
        ]
        return AnswerResult(
            answer=answer, status=status, confidence=round(state.confidence, 6), evidence_references=tuple(state.evidence_items),
            hop_count=len(state.attempted_queries), stop_reason=stop_reason, effective_budget=state.request.budget,
            trace_summary=tuple(state.trace) if state.request.mode.value == "diagnostic" else tuple(state.trace[-1:]),
            errors=errors, answer_diagnostics=result_diagnostics,
            answer_attempt_permitted=permitted,
        )

    def _abstention(self, question: str, reason: str) -> str:
        return f"I do not have sufficient evidence to answer this question ({reason})."
