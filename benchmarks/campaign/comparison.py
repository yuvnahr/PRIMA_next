"""Strict paired validation and benchmark-native paired statistics."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from affect.taxonomies.goemotions import LABELS
from benchmarks.campaign.config import ComparisonConfig
from benchmarks.common.contracts import BenchmarkManifest
from benchmarks.goemotions.metrics import evaluate
from benchmarks.hotpotqa.evaluate import score_hotpot_record
from benchmarks.locomo.evaluate import f1_score, normalized_exact_match


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
    del left_result, right_result
    left = BenchmarkManifest.model_validate_json(left_manifest.read_text(encoding="utf-8"))
    right = BenchmarkManifest.model_validate_json(right_manifest.read_text(encoding="utf-8"))
    checks = {
        "model": left.model == right.model,
        "revision": left.model_revision == right.model_revision or (
            left.model_revision is None and right.model_revision is None and revision is not None
        ),
        "dataset_hash": left.dataset_hash == right.dataset_hash,
        "benchmark": left.benchmark.name == right.benchmark.name and left.benchmark.mode == right.benchmark.mode,
        "context_variant": _context_variant(left) == _context_variant(right),
        "headline_eligibility": _headline_eligible(left) and _headline_eligible(right),
        "selected_ids_and_order": left.selected_ids == right.selected_ids,
        "seed": left.seed == right.seed,
        "generation_config": left.generation_config == right.generation_config,
        "context_budget": left_context_budget == right_context_budget,
        "retry_policy": left.generation_config.get("retries") == right.generation_config.get("retries"),
    }
    failed_checks = [name for name, valid in checks.items() if not valid]
    if failed_checks:
        raise ValueError(f"Comparison {spec.id!r} refused; paired invariants differ: {', '.join(failed_checks)}")
    left_cases, right_cases = _cases(left_manifest), _cases(right_manifest)
    missing = [case_id for case_id in left.selected_ids if case_id not in left_cases or case_id not in right_cases]
    if missing:
        raise ValueError(f"Comparison {spec.id!r} has missing checkpoint IDs: {missing}")

    excluded = {"failed": 0, "parse_failed": 0, "unscored": 0}
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for case_id in left.selected_ids:
        left_case, right_case = left_cases[case_id], right_cases[case_id]
        category = _excluded_category(left_case, right_case)
        if category:
            excluded[category] += 1
        else:
            paired.append((left_case["record"], right_case["record"]))
    if not paired:
        raise ValueError(f"Comparison {spec.id!r} has no paired scored cases")

    metric_name = spec.metric.rsplit(".", 1)[-1]
    left_value = _benchmark_metric(left.benchmark.name, metric_name, [item[0] for item in paired])
    right_value = _benchmark_metric(right.benchmark.name, metric_name, [item[1] for item in paired])
    case_deltas = [
        _case_score(left.benchmark.name, metric_name, right_record)
        - _case_score(left.benchmark.name, metric_name, left_record)
        for left_record, right_record in paired
    ]
    return {
        "status": "valid",
        "checks": checks,
        "metric": spec.metric,
        "left": left_value,
        "right": right_value,
        "delta": right_value - left_value,
        "paired_scored": len(paired),
        "excluded": excluded,
        "corrected": sum(delta > 0 for delta in case_deltas),
        "regressed": sum(delta < 0 for delta in case_deltas),
        "unchanged": sum(delta == 0 for delta in case_deltas),
        "bootstrap_95_ci": _bootstrap_metric(
            left.benchmark.name, metric_name, paired, spec.bootstrap_samples, left.seed or 0
        ),
    }


def _cases(manifest_path: Path) -> dict[str, dict[str, Any]]:
    checkpoint = manifest_path.parent / "checkpoints" / "records.jsonl"
    cases: dict[str, dict[str, Any]] = {}
    if not checkpoint.is_file():
        return cases
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        case_id = str(row["case_id"])
        if case_id in cases:
            raise ValueError(f"Duplicate checkpoint case ID: {case_id}")
        prediction = row.get("prediction") or {}
        diagnostics = prediction.get("diagnostics", {})
        record = next(
            (diagnostics[key] for key in ("goemotions_record", "hotpot_record", "locomo_record") if isinstance(diagnostics.get(key), dict)),
            None,
        )
        cases[case_id] = {"status": row.get("status"), "record": record, "failure": row.get("failure") or {}}
    return cases


def _excluded_category(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    if left["status"] != "complete" or right["status"] != "complete":
        return "failed"
    records: list[dict[str, Any]] = []
    for value in (left.get("record"), right.get("record")):
        if not isinstance(value, dict):
            return "unscored"
        records.append(value)
    for record in records:
        if record.get("parse_error") is not None or record.get("failure_category") == "ANSWER_PARSE_FAILURE":
            return "parse_failed"
        if record.get("scored") is False:
            return "unscored"
    return None


def _benchmark_metric(benchmark: str, metric: str, records: list[dict[str, Any]]) -> float:
    if benchmark == "GoEmotions":
        values = evaluate(
            [frozenset(record.get("gold_labels", ())) for record in records],
            [frozenset(record.get("predicted_labels", ())) for record in records],
            list(LABELS),
        )
        if metric not in values or not isinstance(values[metric], (int, float)):
            raise ValueError(f"Unsupported GoEmotions paired metric: {metric}")
        return float(values[metric])
    scores = [_case_score(benchmark, metric, record) for record in records]
    return sum(scores) / len(scores)


def _case_score(benchmark: str, metric: str, record: dict[str, Any]) -> float:
    if benchmark == "HotpotQA":
        scores = score_hotpot_record(record)
        if metric not in scores:
            raise ValueError(f"Unsupported HotpotQA paired metric: {metric}")
        return float(scores[metric])
    if benchmark == "LoCoMo":
        prediction, expected = str(record.get("prediction", "")), str(record.get("expected_answer", ""))
        if metric == "exact_match":
            return float(normalized_exact_match(prediction, expected))
        if metric == "f1":
            return float(f1_score(prediction, expected))
        raise ValueError(f"Unsupported LoCoMo paired metric: {metric}")
    if benchmark == "GoEmotions":
        gold, predicted = set(record.get("gold_labels", ())), set(record.get("predicted_labels", ()))
        return 2 * len(gold & predicted) / (len(gold) + len(predicted)) if gold or predicted else 1.0
    raise ValueError(f"Unsupported paired benchmark: {benchmark}")


def _bootstrap_metric(
    benchmark: str,
    metric: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    samples: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)  # noqa: S311  # nosec B311 - deterministic confidence interval
    deltas = []
    for _ in range(samples):
        chosen = [rng.choice(pairs) for _ in pairs]
        left = _benchmark_metric(benchmark, metric, [item[0] for item in chosen])
        right = _benchmark_metric(benchmark, metric, [item[1] for item in chosen])
        deltas.append(right - left)
    deltas.sort()
    return [deltas[int(0.025 * (samples - 1))], deltas[int(0.975 * (samples - 1))]]


def _context_variant(manifest: BenchmarkManifest) -> tuple[Any, ...]:
    config = manifest.benchmark_config
    if manifest.benchmark.name == "HotpotQA":
        return (config.get("context_mode"), config.get("context_source"))
    if manifest.benchmark.name == "LoCoMo":
        return (config.get("ingestion_policy"), config.get("dataset_scope"))
    return (config.get("split"), config.get("dataset_scope"))


def _headline_eligible(manifest: BenchmarkManifest) -> bool:
    return manifest.active_capabilities.get("headline_eligible", True) is True
