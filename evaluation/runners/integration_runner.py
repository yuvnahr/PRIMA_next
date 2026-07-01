"""Integration runner scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from runtime.prima_runtime import PrimaRuntime
from runtime.runtime_result import RuntimeResult


@dataclass(slots=True)
class IntegrationRunner:
    """Run a small runtime integration probe."""

    runtime: PrimaRuntime

    def run_once(self, user_input: str) -> RuntimeResult:
        """Execute one runtime integration turn."""
        return self.runtime.process(user_input)
