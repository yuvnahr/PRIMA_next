"""Strict paired validation and benchmark-native paired statistics."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from affect.taxonomies.goemotions import LABELS
from benchmarks.campaign.config import ComparisonConfig
from benchmarks.common.artifacts import read_json, read_jsonl
from benchmarks.common.contracts import BenchmarkManifest
from benchmarks.goemotions.metrics import evaluate
from benchmarks.hotpotqa.evaluate import score_hotpot_record
from benchmarks.locomo.evaluate import (
    answerable_population,
    answerable_score,
    f1_score,
    normalized_exact_match,
    paired_conversation_interval,
    validate_pair_identity,
)
from benchmarks.locomo.official_scoring import official_locomo_score, scorer_identity


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
    left = BenchmarkManifest.model_validate(read_json(left_manifest))
    right = BenchmarkManifest.model_validate(read_json(right_manifest))
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
    if left.benchmark.name == "LoCoMo":
        if left.source_fingerprint != right.source_fingerprint or left.git_commit != right.git_commit:
            raise ValueError("Comparison refused; LoCoMo source identities differ")
        return _locomo_comparison(spec, left, checks, left_cases, right_cases, excluded)
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


def _locomo_comparison(
    spec: ComparisonConfig, manifest: BenchmarkManifest, checks: dict[str, bool],
    left_cases: dict[str, dict[str, Any]], right_cases: dict[str, dict[str, Any]], excluded: dict[str, int],
) -> dict[str, Any]:
    metric = spec.metric.rsplit(".", 1)[-1]
    metric = {"f1": "legacy_squad_f1", "exact_match": "legacy_normalized_exact_match"}.get(metric, metric)
    if metric not in {"legacy_squad_f1", "legacy_normalized_exact_match",
                       "official_locomo_f1", "answerable_official_f1", "adversarial_abstention_accuracy"}:
        raise ValueError(f"Unsupported LoCoMo paired metric: {metric}")
    observations = []
    point_pairs = []
    incomplete_ids = []
    metric_ineligible = 0
    unknown_categories = 0
    answerable = 0
    populations: tuple[list[Any], list[Any]] = ([], [])
    failures = {"left": dict.fromkeys(excluded, 0), "right": dict.fromkeys(excluded, 0)}
    for case_id in manifest.selected_ids:
        left_case, right_case = left_cases[case_id], right_cases[case_id]
        a, b = left_case.get("record"), right_case.get("record")
        for row in (a, b):
            if row is not None and row.get("case_id") != case_id:
                raise ValueError("LoCoMo native record ID differs from checkpoint case ID")
        category = str((a or b or {}).get("category"))
        if category == "None":
            unknown_categories += 1
        elif category != "5":
            answerable += 1
        modern = False
        if (a is not None and b is not None and a.get("conversation_id") and b.get("conversation_id")
            and a.get("category") is not None and b.get("category") is not None):
            validate_pair_identity(a, b)
            modern = True
        elif a is not None and b is not None and a.get("expected_answer") != b.get("expected_answer"):
            raise ValueError("Paired source/question identities differ")
        for side, case, population in zip(("left", "right"), (left_case, right_case), populations, strict=True):
            reason = _excluded_category(case, case)
            if reason:
                failures[side][reason] += 1
            elif case["record"] is not None:
                population.append(case["record"])
        if _excluded_category(left_case, right_case):
            continue
        if (metric == "answerable_official_f1" and category not in {"1", "2", "3", "4"}) or (
            metric == "adversarial_abstention_accuracy" and category != "5"
        ) or (metric == "official_locomo_f1" and category not in {"1", "2", "3", "4", "5"}):
            metric_ineligible += 1
            continue
        if a is None or b is None:
            raise ValueError("Missing native records in a scored LoCoMo pair")
        if metric == "answerable_official_f1":
            av, bv = answerable_score(a, "official_locomo_f1"), answerable_score(b, "official_locomo_f1")
            if av is None or bv is None:
                metric_ineligible += 1
                continue
        else:
            av, bv = _case_score("LoCoMo", metric, a), _case_score("LoCoMo", metric, b)
        point_pairs.append((av, bv))
        if modern:
            observations.append((case_id, a["conversation_id"], av, bv))
        else:
            incomplete_ids.append(case_id)
    inference = paired_conversation_interval(observations, samples=spec.bootstrap_samples, seed=manifest.seed or 0)
    interval = inference["bootstrap_95_ci"] if inference and not incomplete_ids else None
    count = len(point_pairs)
    left = sum(a for a, _ in point_pairs) / count if count else None
    right = sum(b for _, b in point_pairs) / count if count else None
    return {
        "status": "valid", "checks": checks, "metric": spec.metric, "metric_definition": metric,
        "left": left, "right": right, "delta": (right - left) if left is not None and right is not None else None,
        "selected_cases": len(manifest.selected_ids), "paired_scored": count,
        "paired_case_ids": [case for case, *_ in observations],
        "missing_inference_ids": incomplete_ids, "sampling_unit": "conversation",
        "independent_units": inference["conversation_units"] if inference else 0,
        "bootstrap_95_ci": interval, "official_scorer": scorer_identity(),
        "excluded": excluded, "exclusions_by_arm": failures, "metric_ineligible": metric_ineligible,
        "answerability_unknown": unknown_categories,
        "answerable_population": {
            side: answerable_population(rows, selected=answerable) if not unknown_categories else None
            for side, rows in zip(("left", "right"), populations, strict=True)
        },
        "corrected": sum(b > a for a, b in point_pairs), "regressed": sum(b < a for a, b in point_pairs),
        "unchanged": sum(a == b for a, b in point_pairs),
    }


def _cases(manifest_path: Path) -> dict[str, dict[str, Any]]:
    checkpoint = manifest_path.parent / "checkpoints" / "records.jsonl"
    cases: dict[str, dict[str, Any]] = {}
    if not checkpoint.is_file():
        return cases
    for row in read_jsonl(checkpoint):
        case_id = str(row["case_id"])
        if case_id in cases:
            raise ValueError(f"Duplicate checkpoint case ID: {case_id}")
        prediction = row.get("prediction") or {}
        failure = row.get("failure") or {}
        diagnostics = prediction.get("diagnostics", {}) or failure.get("details", {})
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
        if record.get("execution_failed"):
            return "failed"
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
        if metric == "legacy_normalized_exact_match":
            return float(normalized_exact_match(prediction, expected))
        if metric == "legacy_squad_f1":
            return float(f1_score(prediction, expected))
        if metric == "adversarial_abstention_accuracy":
            return float(record.get("outcome") == "abstained")
        if metric == "official_locomo_f1":
            return official_locomo_score(prediction, record.get("reference_metadata", {}).get("answer", expected), record["category"])
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
