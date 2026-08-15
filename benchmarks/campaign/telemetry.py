"""Best-effort host/GPU telemetry that never controls campaign success."""

from __future__ import annotations

from typing import Any


def collect_host_telemetry(include_gpu: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"available": True}
    try:
        import psutil  # type: ignore[import-untyped]

        process = psutil.Process()
        result.update(
            cpu_percent=psutil.cpu_percent(interval=None),
            process_cpu_percent=process.cpu_percent(interval=None),
            ram_used_bytes=psutil.virtual_memory().used,
            process_rss_bytes=process.memory_info().rss,
        )
    except Exception as exc:  # telemetry is explicitly non-fatal
        result.update(available=False, error=str(exc))
    if include_gpu:
        result["gpu"] = _gpu_telemetry()
    return result


def _gpu_telemetry() -> dict[str, Any]:
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        devices = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            devices.append(
                {
                    "index": index,
                    "utilization_percent": utilization.gpu,
                    "memory_used_bytes": memory.used,
                    "memory_total_bytes": memory.total,
                }
            )
        return {"available": True, "devices": devices}
    except Exception as exc:  # optional telemetry must not crash a campaign
        return {"available": False, "error": str(exc), "devices": []}
