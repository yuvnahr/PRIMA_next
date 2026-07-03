"""LoCoMo runner entry point."""

from __future__ import annotations

from benchmarks.common.runner import GenericBenchmarkRunner
from benchmarks.common.utils import configure_benchmark_logger
from benchmarks.locomo.config import LOG_LEVEL, LOG_PATH


class LoCoMoRunner(GenericBenchmarkRunner):
    """LoCoMo runner configured with benchmark-local logging."""

    def __init__(self) -> None:
        super().__init__(configure_benchmark_logger("benchmarks.locomo.runner", LOG_PATH, LOG_LEVEL))
