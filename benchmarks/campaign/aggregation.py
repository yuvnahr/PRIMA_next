"""Validated campaign aggregation and compact Markdown reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.common import atomic_write_json
from security.redaction import redact


def write_aggregate(
    root: Path,
    campaign_id: str,
    modes: dict[str, Any],
    comparisons: dict[str, Any],
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    included = {
        mode_id: {"status": item.status, "summary": item.summary, "child_manifest": item.child_manifest}
        for mode_id, item in modes.items()
        if item.status in {"complete", "partial"}
    }
    safe_comparisons = redact(comparisons, redact_prompts=False)
    safe_telemetry = redact(telemetry, redact_prompts=False)
    aggregate = {
        "schema_version": "1.0",
        "campaign_id": campaign_id,
        "included_modes": included,
        "comparisons": safe_comparisons,
        "telemetry": safe_telemetry,
    }
    atomic_write_json(root / "aggregate.json", aggregate)
    lines = [f"# Campaign {campaign_id}", "", "## Benchmark modes", ""]
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
    lines.extend(["", "## Inference telemetry", "", f"- Requests: {safe_telemetry.get('request_attempts', 0)}"])
    lines.append(f"- Maximum concurrent inference requests: {safe_telemetry.get('max_active_requests', 0)}")
    lines.append(
        f"- Prompt/completion tokens: {safe_telemetry.get('prompt_tokens', 0)}/"
        f"{safe_telemetry.get('completion_tokens', 0)}"
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate
