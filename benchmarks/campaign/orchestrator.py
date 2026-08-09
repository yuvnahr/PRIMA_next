"""Top-level lifecycle for one resumable three-benchmark campaign."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from benchmarks.campaign.aggregation import write_aggregate
from benchmarks.campaign.comparison import compare
from benchmarks.campaign.config import BenchmarkModeConfig, CampaignConfig
from benchmarks.campaign.manifest import CampaignManifestStore
from benchmarks.campaign.preflight import run_preflight
from benchmarks.campaign.provider_session import SharedProviderSession
from benchmarks.campaign.registry import child_manifest_path, execute_mode
from benchmarks.campaign.scheduler import schedule_modes
from benchmarks.campaign.telemetry import collect_host_telemetry
from benchmarks.common.contracts import BenchmarkManifest, RunStatus


def run_campaign(config: CampaignConfig, *, resume: bool = False) -> dict[str, Any]:
    session = SharedProviderSession(config.provider, config.scheduler.max_gpu_requests)
    preflight = run_preflight(config, session)
    store = CampaignManifestStore(config.output_root)
    manifest = store.initialize(config, preflight, resume=resume)
    host_before = collect_host_telemetry(config.telemetry.gpu) if config.telemetry.enabled else {}
    results: dict[str, dict[str, Any]] = {}
    pending = [mode for mode in config.benchmarks if manifest.modes[mode.id].status != "complete"]

    def run_one(mode: BenchmarkModeConfig) -> dict[str, Any]:
        mode_root = config.output_root / "modes" / mode.id
        child = child_manifest_path(mode, mode_root)
        store.update_mode(mode.id, status="running", error=None)
        try:
            result = execute_mode(config, mode, session, mode_root, resume=resume and child.is_file())
            child_status = _child_status(child)
            if child_status is not RunStatus.COMPLETE:
                raise ValueError(f"Benchmark {mode.id} returned without a complete validated child manifest")
            results[mode.id] = result
            store.update_mode(
                mode.id,
                status="complete",
                child_manifest=str(child),
                summary=_compact_summary(result),
            )
            return result
        except Exception as exc:
            valid_child = _child_status(child)
            state = "partial" if valid_child is not None else "failed"
            store.update_mode(
                mode.id,
                status=state,
                child_manifest=str(child) if valid_child is not None else None,
                error=str(exc),
            )
            if not config.failure_policy.continue_benchmark_failures:
                raise
            return {"error": str(exc)}

    schedule_modes(pending, config.scheduler, run_one)
    current = store.read()
    for mode in config.benchmarks:
        if mode.id not in results and current.modes[mode.id].status == "complete":
            results[mode.id] = current.modes[mode.id].summary

    comparisons: dict[str, Any] = {}
    modes_by_id = {mode.id: mode for mode in config.benchmarks}
    for spec in config.comparisons:
        left_state, right_state = current.modes[spec.left], current.modes[spec.right]
        try:
            if left_state.status != "complete" or right_state.status != "complete":
                raise ValueError("both paired modes must be complete")
            comparisons[spec.id] = compare(
                spec,
                Path(left_state.child_manifest or ""),
                Path(right_state.child_manifest or ""),
                results[spec.left],
                results[spec.right],
                config.provider.revision,
                modes_by_id[spec.left].context_budget,
                modes_by_id[spec.right].context_budget,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            comparisons[spec.id] = {"status": "refused", "error": str(exc)}
    try:
        telemetry = session.telemetry()
    except Exception as exc:  # telemetry is explicitly non-fatal
        telemetry = {"available": False, "error": str(exc)}
    telemetry["host_before"] = host_before
    telemetry["host_after"] = collect_host_telemetry(config.telemetry.gpu) if config.telemetry.enabled else {}
    telemetry["gpu_peak_observed_memory_bytes"] = _observed_gpu_peak(
        telemetry["host_before"], telemetry["host_after"]
    )
    current = store.read()
    write_aggregate(config.output_root, current.campaign_id, current.modes, comparisons, telemetry)
    statuses = {item.status for item in current.modes.values()}
    status: Literal["complete", "partial"] = (
        "complete"
        if statuses == {"complete"} and all(x.get("status") != "refused" for x in comparisons.values())
        else "partial"
    )
    final = store.finalize(status, comparisons=comparisons, telemetry=telemetry)
    return {
        "campaign_id": final.campaign_id,
        "status": final.status,
        "output_root": str(config.output_root),
        "modes": {key: value.model_dump(mode="json") for key, value in final.modes.items()},
        "comparisons": comparisons,
        "telemetry": telemetry,
    }


def _compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"artifacts"} and (key != "metrics" or isinstance(value, dict))
    }


def _observed_gpu_peak(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    peak: dict[str, int] = {}
    for snapshot in (before, after):
        gpu = snapshot.get("gpu", {}) if isinstance(snapshot, dict) else {}
        for device in gpu.get("devices", ()) if isinstance(gpu, dict) else ():
            key = str(device.get("index"))
            peak[key] = max(peak.get(key, 0), int(device.get("memory_used_bytes", 0)))
    return peak


def _child_status(path: Path) -> RunStatus | None:
    if not path.is_file():
        return None
    try:
        return BenchmarkManifest.model_validate_json(path.read_text(encoding="utf-8")).status
    except (OSError, ValueError):
        return None
