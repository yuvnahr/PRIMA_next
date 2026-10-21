"""Durable monotonic attempt durations; resumed calendar downtime is not work."""

from __future__ import annotations

import math
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchmarks.common.artifacts import append_jsonl, read_jsonl


class AttemptTimer:
    def __init__(self, path: Path, *, started: float | None = None) -> None:
        self.path = path
        self.started = time.perf_counter() if started is None else started
        self.attempt_id = str(uuid4())
        self._lock = threading.Lock()
        self.record("start")

    def record(self, phase: str, *, finished: bool = False) -> None:
        with self._lock:
            append_jsonl(self.path, {
                "attempt_id": self.attempt_id, "phase": phase,
                "elapsed_seconds": time.perf_counter() - self.started,
                "observed_at": datetime.now(UTC).isoformat(), "finished": finished,
            })

    def summary(self) -> dict[str, Any]:
        attempts: dict[str, dict[str, Any]] = {}
        phases: dict[str, float] = {}
        for row in read_jsonl(self.path):
            elapsed = row["elapsed_seconds"]
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Invalid active attempt duration")
            if not isinstance(row.get("finished"), bool) or not isinstance(row.get("attempt_id"), str) or not row["attempt_id"]:
                raise ValueError("Invalid active attempt identity or closure marker")
            previous = attempts.get(row["attempt_id"])
            if previous and elapsed < previous["elapsed_seconds"]:
                raise ValueError("Attempt duration moved backwards")
            phases[row["phase"]] = phases.get(row["phase"], 0.0) + elapsed - (previous["elapsed_seconds"] if previous else 0)
            attempts[row["attempt_id"]] = row
        complete = all(row["finished"] for row in attempts.values())
        starts = [datetime.fromisoformat(row["observed_at"]) - timedelta(seconds=row["elapsed_seconds"]) for row in attempts.values()]
        finishes = [datetime.fromisoformat(row["observed_at"]) for row in attempts.values()]
        return {
            "active_wall_seconds": sum(row["elapsed_seconds"] for row in attempts.values()),
            "complete": complete, "attempt_count": len(attempts), "phase_seconds": phases,
            "calendar_seconds": (max(finishes) - min(starts)).total_seconds(),
            "measurement": "sum of monotonic active attempts; open attempts are observed lower bounds; calendar downtime excluded",
        }
