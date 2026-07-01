"""Stress runner scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from runtime.prima_runtime import PrimaRuntime
from runtime.runtime_result import RuntimeResult


@dataclass(slots=True)
class StressRunner:
    """Repeated runtime execution scaffold."""

    runtime: PrimaRuntime

    def run(self, user_input: str, iterations: int = 1) -> list[RuntimeResult]:
        """Run the same input repeatedly for future stress tests."""
        return [self.runtime.process(user_input) for _ in range(max(0, iterations))]
