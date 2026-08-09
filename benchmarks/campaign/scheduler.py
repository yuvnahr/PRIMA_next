"""Campaign scheduling with a single shared inference queue."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from benchmarks.campaign.config import BenchmarkModeConfig, SchedulerConfig


def schedule_modes(
    modes: list[BenchmarkModeConfig],
    config: SchedulerConfig,
    execute: Callable[[BenchmarkModeConfig], Any],
) -> dict[str, Any]:
    if config.mode == "sequential":
        return {mode.id: execute(mode) for mode in modes}
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=min(config.cpu_workers, len(modes))) as executor:
        futures = {executor.submit(execute, mode): mode.id for mode in modes}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results
