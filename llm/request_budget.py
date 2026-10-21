"""Shared request accounting across evidence acquisition and provider retries."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import monotonic
from typing import Any


class RequestBudgetExceeded(RuntimeError):
    """A request cannot start another provider call."""


@dataclass(slots=True)
class RequestBudget:
    max_llm_calls: int = 4
    time_budget_seconds: float = 15.0
    answer_reserve_seconds: float = 3.0
    started_at: float = field(default_factory=monotonic)
    llm_calls: int = 0
    retrieval_calls: int = 0
    bounded_answer_attempts: int = 0

    def __post_init__(self) -> None:
        if (self.max_llm_calls < 0 or not math.isfinite(self.time_budget_seconds) or self.time_budget_seconds <= 0
                or not math.isfinite(self.answer_reserve_seconds) or self.answer_reserve_seconds < 0):
            raise ValueError("Request call/time limits must be finite and nonnegative, with a positive deadline")

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.time_budget_seconds - (monotonic() - self.started_at))

    @property
    def retrieval_time_exhausted(self) -> bool:
        reserve = min(self.answer_reserve_seconds, self.time_budget_seconds * 0.2)
        return self.remaining_seconds <= reserve

    def denial_reason(self) -> str:
        if self.remaining_seconds <= 0:
            return "time_budget"
        if self.llm_calls >= self.max_llm_calls:
            return "max_llm_calls"
        return ""

    def reserve_call(self) -> None:
        reason = self.denial_reason()
        if reason:
            raise RequestBudgetExceeded(reason)
        self.llm_calls += 1

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "1.0", "max_llm_calls": self.max_llm_calls, "llm_calls": self.llm_calls,
                "retrieval_calls": self.retrieval_calls, "time_budget_seconds": self.time_budget_seconds,
                "answer_reserve_seconds": self.answer_reserve_seconds,
                "remaining_seconds": self.remaining_seconds, "bounded_answer_attempts": self.bounded_answer_attempts}


_PREPAID_CALL: ContextVar[tuple[RequestBudget, bool] | None] = ContextVar("prima_prepaid_call", default=None)


@contextmanager
def prepaid_call(budget: RequestBudget) -> Iterator[None]:
    """The caller reserved the first call; the client charges every retry."""
    token = _PREPAID_CALL.set((budget, True))
    try:
        yield
    finally:
        _PREPAID_CALL.reset(token)


def before_provider_call() -> RequestBudget | None:
    active = _PREPAID_CALL.get()
    if active is None:
        return None
    budget, prepaid = active
    if budget.remaining_seconds <= 0:
        raise RequestBudgetExceeded("time_budget")
    if prepaid:
        _PREPAID_CALL.set((budget, False))
    else:
        budget.reserve_call()
    return budget
