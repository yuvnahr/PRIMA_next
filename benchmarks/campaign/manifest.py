"""Crash-safe root campaign manifest and resume validation."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.campaign.config import CampaignConfig
from benchmarks.common import atomic_write_json


class CampaignModeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["pending", "running", "complete", "partial", "failed"] = "pending"
    child_manifest: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class CampaignManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"] = "1.0"
    campaign_id: str
    config_hash: str
    status: Literal["partial", "complete", "failed"] = "partial"
    provider: dict[str, Any]
    preflight: dict[str, Any]
    modes: dict[str, CampaignModeRecord]
    comparisons: dict[str, Any] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class CampaignManifestStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "manifest.json"
        self._lock = threading.Lock()

    def initialize(self, config: CampaignConfig, preflight: dict[str, Any], *, resume: bool) -> CampaignManifest:
        fingerprint = config_hash(config)
        if resume:
            if not self.path.is_file():
                raise ValueError(f"Cannot resume without campaign manifest: {self.path}")
            manifest = self.read()
            if manifest.config_hash != fingerprint:
                raise ValueError("Campaign configuration changed; resume refused")
            manifest = manifest.model_copy(update={"status": "partial", "updated_at": _now()})
        else:
            if self.path.exists():
                raise ValueError(f"Campaign output already exists; use --resume or a new output_root: {self.root}")
            now = _now()
            manifest = CampaignManifest(
                campaign_id=f"campaign-{uuid4()}",
                config_hash=fingerprint,
                provider={
                    "kind": config.provider.kind,
                    "model": config.provider.model,
                    "revision": config.provider.revision,
                },
                preflight=preflight,
                modes={mode.id: CampaignModeRecord() for mode in config.benchmarks},
                created_at=now,
                updated_at=now,
            )
        self.write(manifest)
        return manifest

    def read(self) -> CampaignManifest:
        return CampaignManifest.model_validate_json(self.path.read_text(encoding="utf-8"))

    def write(self, manifest: CampaignManifest) -> None:
        atomic_write_json(self.path, manifest.model_dump(mode="json"))

    def update_mode(self, mode_id: str, **values: Any) -> CampaignManifest:
        with self._lock:
            manifest = self.read()
            modes = dict(manifest.modes)
            modes[mode_id] = modes[mode_id].model_copy(update=values)
            updated = manifest.model_copy(update={"modes": modes, "updated_at": _now()})
            self.write(updated)
            return updated

    def finalize(self, status: Literal["partial", "complete", "failed"], **values: Any) -> CampaignManifest:
        with self._lock:
            manifest = self.read().model_copy(update={"status": status, "updated_at": _now(), **values})
            self.write(manifest)
            return manifest


def config_hash(config: CampaignConfig) -> str:
    payload = config.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)
