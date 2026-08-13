"""Crash-safe benchmark artifact storage and exact resume validation."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.common.contracts import (
    BenchmarkManifest,
    BenchmarkSummary,
    CampaignReference,
    CheckpointRecord,
    FailureRecord,
    PredictionRecord,
    RunStatus,
    utc_now,
)
from security.redaction import redact


class ResumeCompatibilityError(ValueError):
    """Existing and requested campaigns cannot be resumed together."""


@dataclass(frozen=True, slots=True)
class ArtifactLayout:
    """Predictable paths for one benchmark campaign."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints" / "records.jsonl"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions" / "records.jsonl"

    @property
    def failures(self) -> Path:
        return self.root / "failures" / "records.jsonl"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"


class BenchmarkArtifactStore:
    """Own atomic manifests/finalization and duplicate-safe append artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.layout = ArtifactLayout(Path(root))
        # ponytail: process-local lock; add an OS file lock only when multi-process writers are adopted.
        self._lock = threading.Lock()
        self._indexes: dict[Path, set[str]] = {}

    def initialize(self, manifest: BenchmarkManifest) -> None:
        """Create a partial campaign manifest before case execution."""

        if manifest.status is RunStatus.COMPLETE:
            raise ValueError("a campaign must initialize as partial, failed, or cancelled")
        atomic_write_json(self.layout.manifest, manifest.to_dict())
        self.reconcile()

    def read_manifest(self) -> BenchmarkManifest:
        """Load and validate the canonical campaign status marker."""

        return BenchmarkManifest.model_validate(read_json(self.layout.manifest))

    def checkpoints(self) -> tuple[CheckpointRecord, ...]:
        """Read every checkpoint strictly; malformed rows are never skipped."""

        return tuple(
            CheckpointRecord.model_validate(row)
            for row in read_jsonl(self.layout.checkpoints)
        )

    def append_checkpoint(self, record: CheckpointRecord) -> bool:
        """Append one case exactly once; return false for an existing case ID."""

        appended = self._append_unique(self.layout.checkpoints, record.case_id, record.to_dict())
        if appended:
            if record.prediction is not None:
                self.append_prediction(record.prediction)
            if record.failure is not None:
                self.append_failure(record.failure)
        return appended

    def append_prediction(self, record: PredictionRecord) -> bool:
        """Append one prediction exactly once."""

        return self._append_unique(self.layout.predictions, record.case_id, record.to_dict())

    def append_failure(self, record: FailureRecord) -> bool:
        """Append one case failure exactly once."""

        key = record.case_id or f"campaign:{record.stage.value}:{record.occurred_at.isoformat()}"
        return self._append_unique(self.layout.failures, key, record.to_dict())

    def mark_status(self, status: RunStatus) -> BenchmarkManifest:
        """Atomically persist partial, failed, or cancelled campaign status."""

        if status is RunStatus.COMPLETE:
            raise ValueError("use finalize() to mark a campaign complete")
        manifest = self.read_manifest().model_copy(update={"status": status, "updated_at": utc_now()})
        atomic_write_json(self.layout.manifest, manifest.to_dict())
        return manifest

    def finalize(self, summary: BenchmarkSummary) -> BenchmarkManifest:
        """Write summary first and the authoritative terminal manifest last."""

        manifest = self.read_manifest()
        if summary.campaign_id != manifest.campaign_id:
            raise ValueError("summary belongs to a different campaign")
        checkpoints = self.checkpoints()
        checkpoint_ids = {
            record.case_id for record in checkpoints if record.status is not RunStatus.PARTIAL
        }
        selected_ids = set(manifest.selected_ids)
        if summary.status is RunStatus.COMPLETE and checkpoint_ids != selected_ids:
            missing = sorted(selected_ids - checkpoint_ids)
            extra = sorted(checkpoint_ids - selected_ids)
            raise ValueError(f"complete finalization requires exact selected IDs; missing={missing}, extra={extra}")
        counts = {
            RunStatus.COMPLETE: summary.completed_cases,
            RunStatus.FAILED: summary.failed_cases,
            RunStatus.CANCELLED: summary.cancelled_cases,
        }
        for status, expected in counts.items():
            actual = sum(record.status is status for record in checkpoints)
            if actual != expected:
                raise ValueError(
                    f"summary {status.value} count {expected} does not match {actual} checkpoints"
                )
        atomic_write_json(self.layout.summary, summary.to_dict())
        terminal = manifest.model_copy(update={"status": summary.status, "updated_at": utc_now()})
        atomic_write_json(self.layout.manifest, terminal.to_dict())
        return terminal

    def _append_unique(self, path: Path, key: str, payload: dict[str, Any]) -> bool:
        with self._lock:
            existing = self._indexes.setdefault(
                path,
                {
                    str(row.get("case_id") or row.get("_record_key", ""))
                    for row in read_jsonl(path)
                },
            )
            if key in existing:
                return False
            append_jsonl(path, payload)
            existing.add(key)
            return True

    def reconcile(self) -> None:
        """Project prediction/failure mirrors deterministically from authoritative checkpoints."""

        checkpoints = self.checkpoints()
        if not checkpoints and not (
            self.layout.predictions.exists() or self.layout.failures.exists()
        ):
            return
        predictions = [record.prediction.to_dict() for record in checkpoints if record.prediction is not None]
        failures = [record.failure.to_dict() for record in checkpoints if record.failure is not None]
        campaign_failures = [
            row for row in read_jsonl(self.layout.failures) if row.get("case_id") is None
        ]
        atomic_write_jsonl(self.layout.predictions, predictions)
        atomic_write_jsonl(self.layout.failures, [*campaign_failures, *failures])
        self._indexes[self.layout.checkpoints] = {record.case_id for record in checkpoints}
        self._indexes[self.layout.predictions] = {record.case_id for record in checkpoints if record.prediction}
        self._indexes[self.layout.failures] = {
            str(row.get("case_id") or row.get("_record_key", ""))
            for row in [*campaign_failures, *failures]
        }


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Replace a JSON file atomically without exposing a truncated destination."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(redact(payload, redact_prompts=False), stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Append one flushed, single-line JSON record."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(redact(payload, redact_prompts=False), sort_keys=True, ensure_ascii=False)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Atomically replace a derived JSONL projection."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(redact(row, redact_prompts=False), sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_json(path: str | Path) -> Any:
    """Read one JSON artifact with an actionable malformed-file error."""

    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON artifact {source}: {exc}") from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL strictly; an interrupted/malformed row invalidates resume."""

    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed checkpoint row {source}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Malformed checkpoint row {source}:{line_number}: expected object")
        rows.append(value)
    return rows


def validate_resume(existing: BenchmarkManifest, current: BenchmarkManifest) -> None:
    """Reject any configuration, source, dataset, selection, or model drift."""

    if existing.manifest_fingerprint == current.manifest_fingerprint:
        return
    ignored = {"manifest_fingerprint", "status", "created_at", "updated_at", "resume_lineage", "campaign_id"}
    old = existing.model_dump(mode="json", exclude=ignored)
    new = current.model_dump(mode="json", exclude=ignored)
    mismatches = sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))
    raise ResumeCompatibilityError(f"Incompatible resume fields: {', '.join(mismatches)}")


def resume_manifest(existing: BenchmarkManifest, current: BenchmarkManifest) -> BenchmarkManifest:
    """Validate exact compatibility and append immutable resume lineage."""

    validate_resume(existing, current)
    reference = CampaignReference(
        campaign_id=existing.campaign_id,
        manifest_fingerprint=existing.manifest_fingerprint,
        status=existing.status,
    )
    return current.model_copy(
        update={
            "status": RunStatus.PARTIAL,
            "updated_at": utc_now(),
            "resume_lineage": (*existing.resume_lineage, reference),
        }
    )
