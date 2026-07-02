"""LoCoMo metric computation over runner outputs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from benchmarks.common.interfaces import BenchmarkEvaluator, RunnerResult
from benchmarks.common.metrics import exact_match, mean


class LoCoMoEvaluator(BenchmarkEvaluator):
    """Compute lightweight LoCoMo metrics from runner results."""

    def evaluate(self, results: Iterable[RunnerResult]) -> dict[str, Any]:
        """Compute metrics without loading data or touching agent internals."""

        items = list(results)
        answerable = [item for item in items if item.expected_answer is not None]
        exact_matches = [
            exact_match(item.response.text, item.expected_answer or "")
            for item in answerable
        ]

        return {
            "total_results": len(items),
            "answerable_results": len(answerable),
            "exact_match": mean(exact_matches),
        }
