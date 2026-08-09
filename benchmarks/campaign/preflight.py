"""Fail-fast campaign capability and resource checks."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from benchmarks.campaign.config import CampaignConfig
from benchmarks.campaign.provider_session import SharedProviderSession
from benchmarks.preflight import capability_report
from llm.generation_config import StructuredOutputMode


def run_preflight(config: CampaignConfig, session: SharedProviderSession) -> dict[str, Any]:
    datasets = {}
    for mode in config.benchmarks:
        if not mode.dataset_path.is_file():
            raise FileNotFoundError(f"Campaign dataset not found for {mode.id}: {mode.dataset_path}")
        hashes = {mode.dataset_path.name: _hash(mode.dataset_path)}
        if mode.benchmark == "goemotions":
            for name in ("train.tsv", "dev.tsv", "test.tsv", "emotions.txt"):
                path = mode.dataset_path.with_name(name)
                if not path.is_file():
                    raise FileNotFoundError(f"GoEmotions campaign file not found: {path}")
                hashes[name] = _hash(path)
        datasets[mode.id] = {"path": str(mode.dataset_path), "hashes": hashes}
        if mode.context_budget > config.provider.context_window:
            raise ValueError(
                f"Mode {mode.id} requires context budget {mode.context_budget}, "
                f"above provider limit {config.provider.context_window}"
            )

    if config.provider.kind != "fake":
        if not config.provider.endpoint:
            raise ValueError("Non-fake campaigns require an explicit provider endpoint")
        url = config.provider.endpoint.rstrip("/") + config.provider.health_path
        try:
            with urllib.request.urlopen(url, timeout=min(5.0, config.provider.timeout_seconds)) as response:  # noqa: S310  # nosec B310
                endpoint = {"url": url, "status": int(getattr(response, "status", 200))}
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise ValueError(f"Provider endpoint unavailable: {url}: {exc}") from exc
    else:
        endpoint = {"url": None, "status": "fake"}

    requires_schema = any(
        mode.benchmark == "goemotions"
        and mode.variant not in {"model_only_zero_shot", "trained_encoder"}
        for mode in config.benchmarks
    )
    schema_available = StructuredOutputMode.JSON_SCHEMA in session.provider.capabilities.structured_output
    if requires_schema and (not config.provider.structured_output or not schema_available):
        raise ValueError("Configured GoEmotions modes require provider JSON-schema output")
    if any(mode.seed is not None for mode in config.benchmarks) and not session.provider.capabilities.seed:
        raise ValueError(f"Provider {config.provider.kind!r} does not support the configured deterministic seeds")

    capabilities = {item["name"]: item for item in capability_report()["capabilities"]}
    requested = sorted({metric for mode in config.benchmarks for metric in mode.optional_metrics})
    for metric in requested:
        if metric not in capabilities:
            raise ValueError(f"Unknown optional metric: {metric}")
        if not capabilities[metric]["available"]:
            raise ValueError(f"Optional metric unavailable: {metric}")

    config.output_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=config.output_root, prefix=".preflight-", delete=True):
            pass
    except OSError as exc:
        raise ValueError(f"Campaign output is not writable: {config.output_root}: {exc}") from exc
    free_gb = shutil.disk_usage(config.output_root).free / (1024**3)
    if free_gb < config.scheduler.min_free_disk_gb:
        raise ValueError(
            f"Campaign requires {config.scheduler.min_free_disk_gb:.3f} GiB free; only {free_gb:.3f} GiB available"
        )

    return {
        "schema_version": "1.0",
        "datasets": datasets,
        "endpoint": endpoint,
        "context_window": config.provider.context_window,
        "structured_output": {"required": requires_schema, "available": schema_available},
        "optional_metrics": {name: capabilities[name] for name in requested},
        "output": {"path": str(config.output_root), "writable": True, "free_disk_gb": free_gb},
        "repository_modes": {mode.id: mode.repository_mode for mode in config.benchmarks},
        "gpu_scheduling": {
            "mode": config.scheduler.mode,
            "max_gpu_requests": config.scheduler.max_gpu_requests,
            "gpu_devices": list(config.scheduler.gpu_devices),
            "api_concurrency_validated": config.scheduler.allow_api_concurrency,
        },
    }


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
