"""Measured Kaggle session guards; no inference or benchmark semantics live here."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.common.artifacts import append_jsonl, atomic_write_json


def require_production_identity(repository_sha: str, model_digest: str, dataset_sha: str) -> None:
    for name, value, length in (
        ("repository SHA", repository_sha, 40), ("model digest", model_digest, 64),
        ("dataset SHA-256", dataset_sha, 64),
    ):
        if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
            raise ValueError(f"Explicit expected {name} is required for production")


def prepare_run(root: Path, identity: dict[str, Any], *, resume: bool, fresh: bool) -> bool:
    """Claim a namespace before launch; return whether a partial campaign can resume."""
    marker = root / "run_identity.json"
    if fresh and root.exists():
        raise ValueError(f"Fresh canary/pilot root already exists: {root}; choose a new namespace")
    if root.exists():
        if not resume or not marker.is_file():
            raise ValueError(f"Existing run requires exact identity and explicit resume: {root}")
        if json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise ValueError("Run identity/config/selected IDs mismatch; resume refused")
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Cannot resume without a campaign manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            raise ValueError("Completed campaign is not a fresh measurement or partial resume")
        return True
    root.mkdir(parents=True)
    atomic_write_json(marker, identity)
    return False


class IncrementalJSONL:
    """Read only appended bytes, retaining incomplete UTF-8/JSON lines until newline."""

    def __init__(self) -> None:
        self.offsets: dict[Path, int] = {}
        self.pending: dict[Path, bytes] = {}
        self.bytes_read = 0

    def poll(self, paths: list[Path], *, final: bool = False) -> list[dict[str, Any]]:
        rows = []
        for path in paths:
            offset = self.offsets.get(path, 0)
            if path.stat().st_size < offset:
                raise ValueError(f"Append-only stream was truncated: {path}")
            with path.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read()
                self.offsets[path] = stream.tell()
            self.bytes_read += len(chunk)
            lines = (self.pending.get(path, b"") + chunk).split(b"\n")
            self.pending[path] = lines.pop()
            if final and self.pending[path].strip():
                raise ValueError(f"Incomplete JSONL tail: {path}")
            for line in lines:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"Expected JSON object: {path}")
                    rows.append(value)
        return rows


class CheckpointProgress:
    """Compact counters keyed by stream and case ID; checkpoint validation stays strict."""

    def __init__(self) -> None:
        self.reader = IncrementalJSONL()
        self.seen: set[tuple[Path, str]] = set()
        self.statuses: Counter[str] = Counter()
        self.tokens = 0
        self.latency_sum = self.latest_latency = 0.0
        self.latency_count = 0

    def poll(self, paths: list[Path], *, final: bool = False) -> int:
        added = 0
        for path in paths:
            for row in self.reader.poll([path], final=final):
                case_id = row.get("case_id")
                key = (path, str(case_id))
                if not case_id or key in self.seen:
                    raise ValueError(f"Missing or duplicate checkpoint ID: {path}:{case_id}")
                self.seen.add(key)
                self.statuses[str(row.get("status"))] += 1
                prediction = row.get("prediction") or {}
                self.tokens += int((prediction.get("tokens") or {}).get("total_tokens") or 0)
                latency = (prediction.get("timing") or {}).get("total_ms")
                if latency is not None:
                    self.latest_latency = float(latency)
                    self.latency_sum += self.latest_latency
                    self.latency_count += 1
                added += 1
        return added


def persist_sample(path: Path, sample: dict[str, Any]) -> None:
    """Flush and fsync each compact sample, including the one that trips a limit."""
    append_jsonl(path, sample)


def resource_guard(sample: dict[str, Any], *, vram_gib: float = 12.0,
                   ram_cap_gb: float = 19.0, reserve_gb: float = 3.0) -> None:
    if not 0 < vram_gib <= 12 or not 0 < reserve_gb < ram_cap_gb <= 19:
        raise ValueError("Resource caps cannot exceed 12 GiB/GPU or 19 GB RAM with 3 GB reserve")
    if reserve_gb < 3:
        raise ValueError("At least 3 GB RAM reserve is required")
    gpus, ram = sample.get("gpus", []), sample.get("ram", {})
    if len(gpus) != 2 or len({gpu["index"] for gpu in gpus}) != 2:
        raise ValueError("Exactly two GPUs must be observed")
    if any(gpu["memory_used_mib"] > vram_gib * 1024 for gpu in gpus):
        raise ValueError("Observed per-GPU VRAM exceeds 12 GiB ceiling")
    total, available = ram.get("total_bytes"), ram.get("available_bytes")
    if not total or available is None:
        raise ValueError("RAM safety measurement is unavailable")
    used = total - available
    # GB is decimal here: the user's 19 GB planning cap is stricter than 19 GiB.
    if available < reserve_gb * 10**9 or used > (ram_cap_gb - reserve_gb) * 10**9:
        raise ValueError("RAM planning cap/reserve violated")
    cgroup_used, cgroup_limit = ram.get("cgroup_current_bytes"), ram.get("cgroup_limit_bytes")
    if cgroup_used is not None and cgroup_limit is not None:
        if cgroup_limit - cgroup_used < reserve_gb * 10**9:
            raise ValueError("Cgroup RAM reserve violated")


def estimate_locomo(events: list[dict[str, Any]], full_counts: dict[str, int], *, workers: int,
                    campaign_wall: float, setup_seconds: float, packaging_seconds: float,
                    safety_margin: float, full_turn_counts: dict[str, int] | None = None) -> dict[str, Any]:
    """Project QA work by conversation; replay each full history exactly once.

    Sampled conversations replay their complete history. With full turn counts,
    extrapolate unsampled history/QA using the slowest observed per-turn/per-question
    rates. Scheduling efficiency is measured, never assumed from worker count.
    """
    if workers not in (1, 2) or not full_counts or any(count <= 0 for count in full_counts.values()):
        raise ValueError("Positive full conversation counts and one/two workers are required")
    measurements = {row["conversation_id"]: row for row in events if row.get("phase") == "conversation"}
    if len(measurements) != sum(row.get("phase") == "conversation" for row in events):
        raise ValueError("Duplicate conversation timing; fresh measurement required")
    if not measurements or not set(measurements) <= set(full_counts):
        raise ValueError("Canary conversation identities do not match the full dataset")
    if full_turn_counts is None and set(measurements) != set(full_counts):
        raise ValueError("Canary must replay every full-dataset conversation")
    if full_turn_counts is not None and (
        set(full_turn_counts) != set(full_counts) or any(turns <= 0 for turns in full_turn_counts.values())
        or len(measurements) < min(3, len(full_counts))
    ):
        raise ValueError("Representative multi-conversation timing and exact full history counts required")
    execution = [row for row in events if row.get("phase") == "execution"]
    finalization = [row for row in events if row.get("phase") == "finalization"]
    if len(execution) != 1 or len(finalization) != 1 or execution[0]["parallel_workers"] != workers:
        raise ValueError("Complete execution/finalization measurements at tested concurrency required")
    values = [campaign_wall, setup_seconds, packaging_seconds, safety_margin,
              execution[0]["seconds"], finalization[0]["seconds"]]
    if any(not math.isfinite(float(value)) or value < 0 for value in values) or safety_margin < 0.2:
        raise ValueError("Finite nonnegative timing and at least 20% safety margin required")
    if campaign_wall <= 0 or packaging_seconds <= 0:
        raise ValueError("Fresh wall time and packaging measurement required")
    replay = qa = measured_work = 0.0
    projected_workers = []
    for cid, row in measurements.items():
        count = row["question_count"]
        history, questions, worker = row["history_seconds"], row["qa_seconds"], row["worker_seconds"]
        if (count <= 0 or count > full_counts[cid] or row["historical_turn_count"] <= 0
                or any(not math.isfinite(value) or value <= 0 for value in (history, questions, worker))
                or worker + 0.01 < history + questions):
            raise ValueError("Incomplete conversation history/QA timing")
        if full_turn_counts is not None and row["historical_turn_count"] != full_turn_counts[cid]:
            raise ValueError("Sampled conversation did not replay its full history")
        replay += history
        projected_qa = questions / count * full_counts[cid]
        qa += projected_qa
        measured_work += worker
        projected_workers.append(worker - questions + projected_qa)
    if full_turn_counts is not None:
        replay_rate = max(row["history_seconds"] / row["historical_turn_count"] for row in measurements.values())
        qa_rate = max(row["qa_seconds"] / row["question_count"] for row in measurements.values())
        overhead = max(max(0.0, row["worker_seconds"] - row["history_seconds"] - row["qa_seconds"])
                       for row in measurements.values())
        for cid in set(full_counts) - set(measurements):
            replay_work, qa_work = full_turn_counts[cid] * replay_rate, full_counts[cid] * qa_rate
            replay += replay_work
            qa += qa_work
            projected_workers.append(replay_work + qa_work + overhead)
    factor = max(1 / workers, execution[0]["seconds"] / measured_work)
    execution_projection = max(sum(projected_workers) * factor, max(projected_workers))
    startup = max(0.0, campaign_wall - execution[0]["seconds"] - finalization[0]["seconds"])
    # Final report/ZIP cost scales by item count; keep a floor, plus safety margin.
    scale = sum(full_counts.values()) / sum(row["question_count"] for row in measurements.values())
    finalize_projection = finalization[0]["seconds"] * max(1.0, scale)
    packaging_projection = packaging_seconds * max(1.0, scale)
    campaign = startup + execution_projection + finalize_projection
    remaining = (campaign + packaging_projection) * (1 + safety_margin)
    return {
        "fresh": True, "workers": workers, "full_items": sum(full_counts.values()),
        "measured_items": sum(row["question_count"] for row in measurements.values()),
        "measured_campaign_seconds": campaign_wall, "measured_schedule_factor": factor,
        "measured_conversations": len(measurements), "full_conversations": len(full_counts),
        "extrapolation": "unsampled conversations use slowest observed replay/QA rates; full sampled histories counted once",
        "historical_replay_seconds": replay, "projected_qa_work_seconds": qa,
        "campaign_startup_seconds": startup, "projected_execution_seconds": execution_projection,
        "projected_finalization_seconds": finalize_projection,
        "projected_packaging_seconds": packaging_projection, "setup_seconds": setup_seconds,
        "safety_margin": safety_margin, "projected_campaign_seconds": remaining,
        "projected_remaining_seconds": remaining, "projected_session_seconds": setup_seconds + remaining,
        "measured_items_per_second": sum(row["question_count"] for row in measurements.values()) / campaign_wall,
    }


def budget_gate(estimate: dict[str, Any], *, elapsed_seconds: float, session_hours: float = 12,
                campaign_hours: float = 10.5) -> dict[str, Any]:
    if not estimate.get("fresh") or not estimate.get("resources_validated"):
        raise ValueError("Full run blocked: fresh complete timing and resource validation required")
    values = [elapsed_seconds, session_hours, campaign_hours,
              estimate.get("projected_campaign_seconds"), estimate.get("projected_remaining_seconds"),
              estimate.get("projected_session_seconds")]
    if any(value is None or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Full run blocked: missing/invalid runtime estimate")
    if not 0 < campaign_hours <= 10.5 or not 0 < session_hours <= 12:
        raise ValueError("Budgets cannot exceed 10.5h campaign / 12h session")
    campaign = estimate["projected_campaign_seconds"]
    session = max(estimate["projected_session_seconds"], elapsed_seconds + estimate["projected_remaining_seconds"])
    shortfall = max(0, campaign - campaign_hours * 3600, session - session_hours * 3600)
    result = {"allowed": shortfall == 0, "projected_campaign_hours": campaign / 3600,
              "projected_session_hours": session / 3600, "shortfall_seconds": shortfall}
    if shortfall:
        raise ValueError(f"Full run blocked: campaign={campaign / 3600:.4f}h, session={session / 3600:.4f}h; "
                         f"shortfall={shortfall:.1f}s against {campaign_hours}h/{session_hours}h budgets")
    return result


def validate_pilot(sequential: list[dict[str, Any]], concurrent: list[dict[str, Any]],
                   selected_ids: list[str], *, resources_validated: bool, max_active_requests: int) -> None:
    """Match fixed predictions and evaluation decisions, excluding timing/random memory IDs."""
    if not resources_validated or max_active_requests != 2:
        raise ValueError("Two-worker pilot did not validate resources and actual overlapping model requests")
    fields = ("prediction", "outcome", "execution_failed", "failure_category", "category",
              "candidate_evidence_ids", "final_evidence_ids", "admitted_evidence_ids",
              "candidate_evidence_recall", "final_evidence_recall", "memory_admission_recall",
              "answer_token_coverage", "memory_growth", "maintenance_complete")
    def index(rows: list[dict[str, Any]]) -> dict[str, Any]:
        ids = [row["case_id"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(selected_ids) or len(ids) != len(selected_ids):
            raise ValueError("Pilot selected/checkpoint IDs do not reconcile")
        if any(row.get("execution_failed") for row in rows):
            raise ValueError("Pilot contains execution failures")
        return {row["case_id"]: {key: row.get(key) for key in fields} for row in rows}
    if index(sequential) != index(concurrent):
        raise ValueError("Concurrency changed predictions or evaluation decisions; retain sequential configuration")
