"""Auditable result ZIPs with canonical checkpoint views and prompt references."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST = "bundle_manifest.json"
SECRET = re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}")


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def _safe(name: str) -> str:
    path = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or path.is_absolute() or any(p in {"..", "."} for p in path.parts) or str(path) != name:
        raise ValueError(f"Unsafe artifact path: {name}")
    return name


def _parse(content: bytes, name: str) -> Any:
    text = content.decode("utf-8")
    if name.endswith(".jsonl"):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"JSONL records must be objects: {name}")
        return rows
    return json.loads(text)


def _native(checkpoints: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [record for row in checkpoints
            if isinstance(record := ((row.get("prediction") or {}).get("diagnostics")
                                     or (row.get("failure") or {}).get("details") or {}).get(key), dict)]


def _view(rows: list[dict[str, Any]], kind: str) -> Any:
    if kind in {"predictions", "failures"}:
        key = "prediction" if kind == "predictions" else "failure"
        return [row[key] for row in rows if row.get(key)]
    key = "locomo_record" if kind == "locomo_questions" else "goemotions_record"
    records = _native(rows, key)
    if kind == "locomo_questions":
        return records
    successful = [row for row in records if not row.get("execution_failed")]
    if kind == "goemotions_raw":
        return {"responses": successful, "failures": [row for row in records if row.get("execution_failed")]}
    fields = {"goemotions_parsed": ("id", "predicted_labels", "parse_error"),
              "goemotions_answers": ("id", "gold_labels", "predicted_labels"),
              "goemotions_prompts": ("id", "prompt")}[kind]
    return [{field: row[field] for field in fields} for row in successful]


def _pack_prompts(value: Any, templates: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [_pack_prompts(item, templates) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in {"prompt", "benchmark_prompt", "system_prompt"} and isinstance(item, str):
            markers = [item.find(marker) for marker in ("\nComment: ", "\n<UNTRUSTED_USER_INPUT>") if marker in item]
            split = min(markers) if markers else 0
            prefix, case_input = item[:split], item[split:]
            template_id = _hash(prefix.encode())
            templates[template_id] = prefix
            result[key] = {"$bundle_prompt": template_id, "input": case_input, "sha256": _hash(item.encode())}
        else:
            result[key] = _pack_prompts(item, templates)
    return result


def _unpack_prompts(value: Any, templates: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [_unpack_prompts(item, templates) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"$bundle_prompt", "input", "sha256"}:
        prompt = templates[value["$bundle_prompt"]] + value["input"]
        if _hash(prompt.encode()) != value["sha256"]:
            raise ValueError("Reconstructed prompt hash mismatch")
        return prompt
    return {key: _unpack_prompts(item, templates) for key, item in value.items()}


def _read(manifest: dict[str, Any], name: str, read: Callable[[str], bytes], seen: tuple[str, ...] = ()) -> bytes:
    _safe(name)
    if name in seen:
        raise ValueError("Cyclic artifact reference")
    entry = manifest["artifacts"][name]
    if "view" in entry:
        rows = _parse(_read(manifest, entry["path"], read, (*seen, name)), entry["path"])
        value = _view(rows, entry["view"])
    else:
        content = read(_safe(entry["path"]))
        if _hash(content) != entry["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {name}")
        if not name.endswith((".json", ".jsonl")):
            return content
        value = _unpack_prompts(_parse(content, name), manifest.get("prompt_templates", {}))
    if _hash(_json(value)) != entry["semantic_sha256"]:
        raise ValueError(f"Artifact reconstruction mismatch: {name}")
    return b"\n".join(_json(row) for row in value) + (b"\n" if value else b"") if name.endswith(".jsonl") else _json(value)


def artifact_bytes(path: Path) -> bytes:
    """Read new extracted bundles or unchanged legacy artifacts without rewriting them."""
    path = path.absolute()
    for root in path.parents:
        marker = root / MANIFEST
        if marker.is_file():
            manifest = json.loads(marker.read_text(encoding="utf-8"))
            name = path.relative_to(root).as_posix()
            if name in manifest["artifacts"]:
                def read_canonical(key: str, base: Path = root) -> bytes:
                    return (base / key).read_bytes()
                try:
                    return _read(manifest, name, read_canonical)
                except (KeyError, FileNotFoundError) as exc:
                    raise ValueError(f"Broken canonical artifact reference: {name}") from exc
    return path.read_bytes()


def create_bundle(root: Path, destination: Path, *, required_files: list[str] | None = None) -> dict[str, Any]:
    """Preserve live artifacts; compact only the exported copy and prove each view."""
    root = root.resolve()
    sources = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink not allowed in result bundle: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (any(p in {"models", "cache", "__pycache__", ".git"} for p in relative.parts)
                or path.suffix.lower() in {".bin", ".pt", ".pth", ".safetensors", ".joblib", ".pkl", ".zip"}
                or path.name.startswith(".env") or path.name == MANIFEST):
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt", ".log", ".yaml", ".yml", ".png", ".svg", ".csv", ".html", ".ipynb"}:
            raise ValueError(f"Unrelated file type in result directory: {relative}")
        name = _safe(relative.as_posix())
        content = path.read_bytes()
        if path.suffix.lower() in {".json", ".jsonl", ".md", ".txt", ".log", ".yaml", ".yml"}:
            if SECRET.search(content.decode("utf-8", errors="replace")):
                raise ValueError(f"Possible secret in artifact: {name}")
        sources[name] = content
    if "manifest.json" not in sources:
        raise ValueError("Campaign manifest missing from bundle")
    parsed = {name: _parse(data, name) for name, data in sources.items() if name.endswith((".json", ".jsonl"))}
    views = {}
    for name, rows in parsed.items():
        if not name.endswith("/checkpoints/records.jsonl"):
            continue
        prefix = name.removesuffix("checkpoints/records.jsonl")
        possible = {"predictions.jsonl": "predictions", "failures.jsonl": "failures",
                    prefix + "predictions/records.jsonl": "predictions", prefix + "failures/records.jsonl": "failures",
                    prefix + "raw/questions.json": "locomo_questions", prefix + "raw/responses.json": "goemotions_raw",
                    prefix + "parsed/answers.json": "goemotions_parsed", prefix + "answers/answers.json": "goemotions_answers",
                    prefix + "prompts/prompts.json": "goemotions_prompts"}
        for target, kind in possible.items():
            if target in parsed:
                try:
                    if _view(rows, kind) == parsed[target]:
                        views[target] = {"path": name, "view": kind}
                except KeyError:
                    pass  # A native format we cannot reconstruct is retained verbatim.
    templates: dict[str, str] = {}
    entries: dict[str, dict[str, Any]] = {}
    stored: dict[str, bytes] = {}
    for name, original in sources.items():
        semantic = {"semantic_sha256": _hash(_json(parsed[name]))} if name in parsed else {}
        if name in views:
            entries[name] = {**views[name], **semantic, "source_sha256": _hash(original)}
            continue
        content = original
        if name in parsed:
            value = _pack_prompts(parsed[name], templates)
            content = b"\n".join(_json(row) for row in value) + (b"\n" if value else b"") if name.endswith(".jsonl") else _json(value)
        digest = _hash(content)
        stored[name] = content
        entries[name] = {"path": name, "sha256": digest, "source_sha256": _hash(original), **semantic}
    manifest = {"schema_version": "1.0", "encoding": "canonical_json_with_verified_prompt_refs",
                "artifacts": entries, "prompt_templates": templates,
                "required_files": required_files or ["manifest.json", "aggregate.json", "report.md", "run_identity.json"], "quality_success": "not_established_by_archive_validation"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("Bundle destination already exists")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in stored.items():
            archive.writestr(name, content)
        archive.writestr(MANIFEST, _json(manifest))
    return validate_bundle(destination)


def validate_bundle(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive paths")
        for item in archive.infolist():
            _safe(item.filename.rstrip("/"))
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Archive symlinks are forbidden")
        if archive.testzip():
            raise ValueError("Archive CRC failure")
        manifest = json.loads(archive.read(MANIFEST)) if MANIFEST in names else None
        if manifest is None:
            artifacts = {name: archive.read(name) for name in names if not name.endswith("/")}
        else:
            if manifest.get("schema_version") != "1.0":
                raise ValueError("Unsupported bundle schema")
            for template_id, prefix in manifest.get("prompt_templates", {}).items():
                if _hash(prefix.encode()) != template_id:
                    raise ValueError("Prompt template hash mismatch")
            expected_physical = {MANIFEST} | {entry["path"] for entry in manifest["artifacts"].values() if "view" not in entry}
            if set(names) != expected_physical:
                raise ValueError("Unexpected or missing physical bundle artifacts")
            artifacts = {name: _read(manifest, name, archive.read) for name in manifest["artifacts"]}
        for name, content in artifacts.items():
            if name.endswith((".json", ".jsonl")):
                _parse(content, name)
            if name.endswith((".json", ".jsonl", ".txt", ".md", ".log")) and SECRET.search(content.decode("utf-8", errors="replace")):
                raise ValueError(f"Possible secret in archive: {name}")
        if "manifest.json" not in artifacts:
            raise ValueError("Campaign manifest missing")
        if manifest is not None:
            for required in manifest["required_files"]:
                if required not in artifacts:
                    raise ValueError(f"Required artifact missing: {required}")
        campaigns = [(name, value) for name, content in artifacts.items()
                     if name.endswith("manifest.json") and isinstance(value := json.loads(content), dict)
                     and isinstance(value.get("modes"), dict)]
        for campaign_name, campaign in campaigns:
            base = PurePosixPath(campaign_name).parent
            for mode in campaign["modes"].values():
                reference = mode.get("child_manifest")
                if not reference:
                    if mode.get("status") == "complete":
                        raise ValueError("Complete campaign mode is missing its child manifest")
                    continue
                _safe(reference)
                reference = str(base / reference)
                child = json.loads(artifacts[reference])
                checkpoints = str(PurePosixPath(reference).parent / "checkpoints/records.jsonl")
                rows = _parse(artifacts[checkpoints], checkpoints)
                from benchmarks.common.contracts import BenchmarkManifest, BenchmarkSummary, CheckpointRecord

                BenchmarkManifest.model_validate(child)
                summary_name = str(PurePosixPath(reference).parent / "summary.json")
                summary = BenchmarkSummary.model_validate(json.loads(artifacts[summary_name]))
                for row in rows:
                    CheckpointRecord.model_validate(row)
                if summary.total_cases != len(child["selected_ids"]):
                    raise ValueError("Summary selected count mismatch")
                counts = {status: sum(row["status"] == status for row in rows) for status in ("complete", "failed", "cancelled")}
                if (summary.completed_cases, summary.failed_cases, summary.cancelled_cases) != (
                        counts["complete"], counts["failed"], counts["cancelled"]):
                    raise ValueError("Summary/checkpoint terminal status counts do not reconcile")
                if any(row["status"] not in counts for row in rows):
                    raise ValueError("Checkpoint contains a nonterminal outcome")
                ids = [row.get("case_id") for row in rows]
                selected = child.get("selected_ids", [])
                if len(ids) != len(set(ids)) or not set(ids) <= set(selected) or len(selected) != len(set(selected)):
                    raise ValueError("Bundle selected/checkpoint IDs do not reconcile")
                if child.get("status") == "complete" and set(ids) != set(selected):
                    raise ValueError("Complete bundle is missing checkpoint IDs")
        return {"valid": True, "archive_sha256": _hash(path.read_bytes()), "logical_artifacts": len(artifacts),
                "stored_files": len(names), "schema_version": "1.0" if manifest else "legacy"}


def require_writable_artifact(path: Path) -> None:
    if any((parent / MANIFEST).is_file() for parent in path.absolute().parents):
        raise ValueError("Exported bundle is immutable; materialize_bundle into a fresh directory before resuming")


def materialize_bundle(archive_path: Path, destination: Path) -> dict[str, Any]:
    """Verify and reconstruct a writable campaign without altering the exported evidence."""
    result = validate_bundle(archive_path)
    if destination.exists():
        raise ValueError("Materialization destination must not exist")
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(MANIFEST)) if MANIFEST in archive.namelist() else None
        names = manifest["artifacts"] if manifest else [name for name in archive.namelist() if not name.endswith("/")]
        for name in names:
            _safe(name)
            content = _read(manifest, name, archive.read) if manifest else archive.read(name)
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        if manifest:
            (destination / "bundle_provenance.json").write_bytes(_json({"archive_sha256": result["archive_sha256"], "export_manifest": manifest}))
    return result
