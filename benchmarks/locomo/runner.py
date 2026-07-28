"""LoCoMo runner entry point."""

from __future__ import annotations

from pathlib import Path

from benchmarks.common.runner import GenericBenchmarkRunner
from benchmarks.common.utils import configure_benchmark_logger
from benchmarks.locomo.config import LOG_LEVEL, LOG_PATH


class LoCoMoRunner(GenericBenchmarkRunner):
    """LoCoMo runner configured with benchmark-local logging."""

    def __init__(self, log_path: str | Path = LOG_PATH) -> None:
        super().__init__(configure_benchmark_logger("benchmarks.locomo.runner", Path(log_path), LOG_LEVEL))
