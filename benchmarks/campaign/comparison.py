"""Strict paired validation and deterministic paired statistics."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from benchmarks.campaign.config import ComparisonConfig
from benchmarks.common.contracts import BenchmarkManifest


def compare(
    spec: ComparisonConfig,
    left_manifest: Path,
    right_manifest: Path,
    left_result: dict[str, Any],
    right_result: dict[str, Any],
    revision: str | None,
    left_context_budget: int,
    right_context_budget: int,
) -> dict[str, Any]:
    left = BenchmarkManifest.model_validate_json(left_manifest.read_text(encoding="utf-8"))
    right = BenchmarkManifest.model_validate_json(right_manifest.read_text(encoding="utf-8"))
    checks = {
        "model": left.model == right.model,
        "revision": left.model_revision == right.model_revision or (
            left.model_revision is None and right.model_revision is None and revision is not None
        ),
        "selected_ids_and_order": left.selected_ids == right.selected_ids,
        "seed": left.seed == right.seed,
        "generation_config": left.generation_config == right.generation_config,
        "context_budget": left_context_budget == right_context_budget,
        "retry_policy": left.generation_config.get("retries") == right.generation_config.get("retries"),
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"Comparison {spec.id!r} refused; paired invariants differ: {', '.join(failed)}")
    left_value = _metric(left_result, spec.metric)
    right_value = _metric(right_result, spec.metric)
    left_scores = _case_scores(left_manifest)
    right_scores = _case_scores(right_manifest)
    common = list(left.selected_ids)
    deltas = [right_scores.get(case_id, 0.0) - left_scores.get(case_id, 0.0) for case_id in common]
    return {
        "status": "valid",
        "checks": checks,
        "metric": spec.metric,
        "left": left_value,
        "right": right_value,
        "delta": right_value - left_value,
        "corrected": sum(delta > 0 for delta in deltas),
        "regressed": sum(delta < 0 for delta in deltas),
        "unchanged": sum(delta == 0 for delta in deltas),
        "bootstrap_95_ci": _bootstrap(deltas, spec.bootstrap_samples, left.seed or 0),
    }


def _metric(result: dict[str, Any], path: str) -> float:
    value: Any = result
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Comparison metric not found: {path}")
        value = value[part]
    if not isinstance(value, (int, float)):
        raise ValueError(f"Comparison metric is not numeric: {path}")
    return float(value)


def _case_scores(manifest_path: Path) -> dict[str, float]:
    checkpoint = manifest_path.parent / "checkpoints" / "records.jsonl"
    scores: dict[str, float] = {}
    if not checkpoint.is_file():
        return scores
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        prediction = row.get("prediction") or {}
        diagnostics = prediction.get("diagnostics", {})
        value: float | bool
        if "goemotions_record" in diagnostics:
            record = diagnostics["goemotions_record"]
            value = set(record.get("gold_labels", ())) == set(record.get("predicted_labels", ()))
        elif "hotpot_record" in diagnostics:
            value = diagnostics["hotpot_record"].get("per_item_scores", {}).get("joint_em", 0.0)
        elif "locomo_record" in diagnostics:
            record = diagnostics["locomo_record"]
            value = _normalized(record.get("prediction", "")) == _normalized(record.get("expected_answer", ""))
        else:
            value = 0.0
        scores[str(row["case_id"])] = float(value)
    return scores


def _bootstrap(values: list[float], samples: int, seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)  # noqa: S311  # nosec B311  # deterministic bootstrap only
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]


def _normalized(value: Any) -> str:
    return " ".join(str(value).casefold().split())
