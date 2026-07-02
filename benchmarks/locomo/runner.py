"""LoCoMo runner entry point."""

from __future__ import annotations

from benchmarks.common.runner import GenericBenchmarkRunner


class LoCoMoRunner(GenericBenchmarkRunner):
    """LoCoMo-specific runner alias using the generic benchmark contract."""
