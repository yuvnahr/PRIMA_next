"""Append-safe HotpotQA checkpoint and resume helpers."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

COMPATIBILITY_FIELDS = ("dataset_fingerprint", "dataset_set", "benchmark_mode", "provider", "model", "reasoning_mode", "top_k", "max_hops", "sampling_strategy", "resolved_seed", "offset", "max_samples", "selected_sample_ids")

def validate_resume(existing: dict[str, Any], current: dict[str, Any]) -> None:
    mismatches = [field for field in COMPATIBILITY_FIELDS if existing.get(field) != current.get(field)]
    if mismatches:
        raise ValueError(f"Incompatible HotpotQA resume manifest fields: {', '.join(mismatches)}")

def read_checkpoint(path: str | Path) -> list[dict[str, Any]]:
    checkpoint = Path(path)
    if not checkpoint.exists(): return []
    rows = []
    for number, line in enumerate(checkpoint.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError as exc: raise ValueError(f"Invalid checkpoint JSONL at line {number}: {exc}") from exc
    return rows

def append_checkpoint(path: str | Path, record: dict[str, Any]) -> None:
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
