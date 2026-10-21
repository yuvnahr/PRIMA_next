"""Validated campaign aggregation and compact Markdown reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.common import atomic_write_json
from benchmarks.common.artifacts import read_jsonl
from benchmarks.common.reporting import classification_report, locomo_report
from security.redaction import redact


def write_aggregate(
    root: Path,
    campaign_id: str,
    modes: dict[str, Any],
    comparisons: dict[str, Any],
    telemetry: dict[str, Any],
    *,
    run_kind: str = "benchmark",
) -> dict[str, Any]:
    included = {
        mode_id: {"status": item.status, "summary": item.summary, "child_manifest": item.child_manifest}
        for mode_id, item in modes.items()
        if item.status in {"complete", "partial"}
    }
    safe_comparisons = redact(comparisons, redact_prompts=False)
    safe_telemetry = redact(telemetry, redact_prompts=False)
    report_semantics = {}
    for mode_id, item in modes.items():
        if not item.child_manifest:
            continue
        child = root / item.child_manifest
        manifest = json.loads(child.read_text(encoding="utf-8"))
        benchmark = manifest["benchmark"]["name"]
        if benchmark not in {"GoEmotions", "LoCoMo"}:
            continue
        key = "goemotions_record" if benchmark == "GoEmotions" else "locomo_record"
        records = []
        checkpoints = read_jsonl(child.parent / "checkpoints/records.jsonl")
        for checkpoint in checkpoints:
            prediction, failure = checkpoint.get("prediction") or {}, checkpoint.get("failure") or {}
            record = (prediction.get("diagnostics") or failure.get("details") or {}).get(key)
            if isinstance(record, dict):
                records.append(record)
        if benchmark == "GoEmotions":
            report_semantics[mode_id] = classification_report(records, item.summary.get("metrics"))
        else:
            replay = child.parent / "raw/conversation_diagnostics.json"
            report_semantics[mode_id] = locomo_report(records, json.loads(replay.read_text(encoding="utf-8")) if replay.is_file() else None)
        report_semantics[mode_id]["selected_id_count"] = len(manifest["selected_ids"])
        report_semantics[mode_id]["checkpoint_count"] = len(checkpoints)
        report_semantics[mode_id]["embedded_record_count"] = len(records)
        report_semantics[mode_id]["unavailable_embedded_records"] = len(checkpoints) - len(records)
    aggregate = {
        "schema_version": "1.0",
        "run_kind": run_kind,
        "reportable": run_kind != "smoke",
        "campaign_id": campaign_id,
        "included_modes": included,
        "comparisons": safe_comparisons,
        "telemetry": safe_telemetry,
        "report_semantics": redact(report_semantics, redact_prompts=False),
    }
    atomic_write_json(root / "aggregate.json", aggregate)
    lines = [f"# Campaign {campaign_id}", "", "## Benchmark modes", ""]
    if run_kind == "smoke":
        lines.insert(2, "PARTIAL SMOKE — non-reportable; full dataset coverage is incomplete.\n")
    lines.extend(
        f"- `{mode_id}`: **{item.status}**"
        + (f" — {redact({'error': item.error}, redact_prompts=False)['error']}" if item.error else "")
        for mode_id, item in modes.items()
    )
    lines.extend(["", "## Paired comparisons", ""])
    if safe_comparisons:
        for name, item in safe_comparisons.items():
            if item.get("status") == "valid":
                lines.append(
                    f"- `{name}`: delta {item['delta']:.6f}; corrected {item['corrected']}, "
                    f"regressed {item['regressed']}; 95% CI {item['bootstrap_95_ci']}"
                )
            else:
                lines.append(f"- `{name}`: **refused** — {item.get('error', 'invalid pair')}")
    else:
        lines.append("- No paired comparisons configured.")
    lines.extend(["", "## Inference telemetry", "", f"- Requests: {safe_telemetry.get('request_attempts', 'unavailable')}"])
    lines.append(f"- Maximum overlapping provider requests: {safe_telemetry.get('max_active_requests', 'unavailable')}")
    lines.append(
        f"- Observed successful uncached prompt/completion tokens: {safe_telemetry.get('prompt_tokens', 'unavailable')}/"
        f"{safe_telemetry.get('completion_tokens', 'unavailable')} (complete usage: {safe_telemetry.get('actual_token_usage_complete', 'unavailable')})"
    )
    lines.extend(["", "## Benchmark report semantics", "", "```json", json.dumps(aggregate["report_semantics"], indent=2), "```"])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate
