"""Conservative routing without benchmark-specific task templates."""

from __future__ import annotations

from reasoning.models import ReasoningMode, ReasoningRequest, Route


class TaskAnalyzer:
    def route(self, request: ReasoningRequest) -> Route:
        if request.mode is ReasoningMode.BYPASS:
            return Route.BYPASS
        if request.mode is ReasoningMode.SINGLE_PASS:
            return Route.SINGLE_PASS
        question = request.question.strip().lower()
        if not question:
            return Route.CLARIFY
        if question in {"hi", "hello", "thanks", "thank you"}:
            return Route.BYPASS
        return Route.ADAPTIVE
