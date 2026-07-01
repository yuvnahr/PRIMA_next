"""Benchmark runner scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class BenchmarkScenario(Protocol):
    """Minimal interface for future benchmark scenarios."""

    name: str

    def run(self) -> dict[str, Any]:
        """Execute the scenario."""


@dataclass(slots=True)
class BenchmarkRunner:
    """Execution framework for future benchmark suites."""

    scenarios: list[BenchmarkScenario] = field(default_factory=list)

    def run(self) -> list[dict[str, Any]]:
        """Run registered benchmark scenarios."""
        return [{"name": scenario.name, "result": scenario.run()} for scenario in self.scenarios]
