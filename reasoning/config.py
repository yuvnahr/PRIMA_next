"""Conservative environment configuration for the reasoning controller."""

from __future__ import annotations

import os

from reasoning.models import ReasoningBudget, ReasoningMode


def reasoning_mode(value: str | None = None) -> ReasoningMode:
    try:
        return ReasoningMode((value or os.getenv("PRIMA_REASONING_MODE", "adaptive")).lower())
    except ValueError:
        return ReasoningMode.SINGLE_PASS


def reasoning_budget(max_hops: int | None = None, max_context_tokens: int | None = None) -> ReasoningBudget:
    return ReasoningBudget(
        max_hops=max(1, int(max_hops or os.getenv("PRIMA_REASONING_MAX_HOPS", "3"))),
        max_retrieval_calls=max(1, int(os.getenv("PRIMA_REASONING_MAX_RETRIEVAL_CALLS", "3"))),
        max_llm_calls=max(1, int(os.getenv("PRIMA_REASONING_MAX_LLM_CALLS", "4"))),
        max_documents=max(1, int(os.getenv("PRIMA_REASONING_MAX_DOCUMENTS", "12"))),
        max_context_tokens=max(128, int(max_context_tokens or os.getenv("PRIMA_REASONING_MAX_CONTEXT_TOKENS", "1600"))),
        time_budget_seconds=max(0.1, float(os.getenv("PRIMA_REASONING_TIME_BUDGET_SECONDS", "15"))),
        no_progress_limit=max(1, int(os.getenv("PRIMA_REASONING_NO_PROGRESS_LIMIT", "1"))),
        max_reflection_interventions=max(0, int(os.getenv("PRIMA_REASONING_MAX_REFLECTION_INTERVENTIONS", "1"))),
        reflection_confidence_threshold=max(0.0, min(1.0, float(os.getenv("PRIMA_REASONING_REFLECTION_CONFIDENCE", "0.6")))),
    )
