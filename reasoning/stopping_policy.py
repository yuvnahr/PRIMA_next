"""Single, centralized continuation policy for the reasoning loop."""

from __future__ import annotations

from reasoning.models import EvidenceState, SufficiencyDecision, SufficiencyStatus


class StoppingPolicy:
    def stop_reason(self, state: EvidenceState, decision: SufficiencyDecision, *, next_query: str | None = None) -> str | None:
        budget = state.request.budget
        if state.shared_budget.remaining_seconds <= 0:
            return "time_budget"
        if decision.status is SufficiencyStatus.SUFFICIENT:
            return "sufficient"
        if decision.status in {SufficiencyStatus.AMBIGUOUS, SufficiencyStatus.CONTRADICTORY, SufficiencyStatus.UNANSWERABLE, SufficiencyStatus.ERROR}:
            return str(decision.status.value)
        if state.shared_budget.retrieval_calls >= budget.max_retrieval_calls:
            return "max_retrieval_calls"
        if len(state.attempted_queries) >= budget.max_hops:
            return "max_hops"
        if state.context_tokens >= budget.max_context_tokens:
            return "context_budget"
        if state.no_progress_hops >= budget.no_progress_limit:
            return "no_new_evidence"
        if state.shared_budget.retrieval_time_exhausted:
            return "answer_time_reserve"
        if next_query and " ".join(next_query.lower().split()) in {" ".join(query.lower().split()) for query in state.attempted_queries}:
            return "duplicate_query"
        return None
