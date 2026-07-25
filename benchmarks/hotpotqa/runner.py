"""Thin HotpotQA runner configuration."""
from __future__ import annotations
from pathlib import Path
from benchmarks.common.runner import GenericBenchmarkRunner
from benchmarks.common.utils import configure_benchmark_logger

class HotpotQARunner(GenericBenchmarkRunner):
    def __init__(self, log_dir: str | Path) -> None:
        super().__init__(configure_benchmark_logger("benchmarks.hotpotqa.runner", Path(log_dir)))
