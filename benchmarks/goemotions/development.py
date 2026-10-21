"""Frozen development populations and predeclared selection; never test tuning."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from affect.taxonomies.goemotions import LABELS
from benchmarks.goemotions.metrics import evaluate, paired_bootstrap_sample_f1
from benchmarks.goemotions.prompts import PROMPT_VARIANTS, build_prompt
from benchmarks.goemotions.schemas import GOEMOTIONS_RESPONSE_SCHEMA
from benchmarks.goemotions.training.data import load_split, validate_splits


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def freeze_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Refuse overwriting an experiment identity, including an internally invalid one."""
    sealed = {**value, "fingerprint": fingerprint(value)}
    if path.exists():
        if read_frozen_json(path) != sealed:
            raise ValueError(f"Frozen identity changed: {path}")
        return sealed
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(sealed, handle, indent=2, sort_keys=True, allow_nan=False)
    return sealed


def read_frozen_json(path: Path, expected: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    observed = value.pop("fingerprint", None)
    if observed != fingerprint(value) or (expected is not None and observed != expected):
        raise ValueError(f"Frozen identity mismatch: {path}")
    return {**value, "fingerprint": observed}


def freeze_development(data_dir: Path, output_dir: Path, generation: dict[str, Any],
                       model_digest: str, *, limit: int = 512, seed: int = 13) -> dict[str, Any]:
    """Freeze split hashes and rarest-label stratification before any model scores."""
    if limit < 1 or not model_digest:
        raise ValueError("Positive dev limit and pinned model digest required")
    identity = validate_splits(data_dir)
    dev = load_split(data_dir, "dev")
    support = Counter(label for row in dev for label in row.labels)
    strata: dict[str, list[Any]] = defaultdict(list)
    for row in dev:
        stratum = min(row.labels, key=lambda label: (support[label], LABELS.index(label)))
        strata[stratum].append(row)
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    for label in sorted(strata):
        strata[label].sort(key=lambda row: row.example_id)
        rng.shuffle(strata[label])
    size = min(limit, len(dev))
    allocations = dict.fromkeys(strata, 0)
    # Largest proportional deficit gives a bounded, deterministic stratified sample.
    for _ in range(size):
        label = max((label for label in strata if allocations[label] < len(strata[label])),
                    key=lambda label: (size * len(strata[label]) / len(dev) - allocations[label], label))
        allocations[label] += 1
    selected = sorted((row for label, rows in strata.items() for row in rows[:allocations[label]]),
                      key=lambda row: row.example_id)
    manifest = freeze_json(output_dir / "dev_manifest.json", {
        "version": 1, "split": "dev", "seed": seed, "algorithm": "rarest_label_proportional_deficit_v1",
        "dataset_sha256": identity["hashes"]["dev"], "dataset_identity": identity,
        "samples": [{"id": row.example_id, "text_sha256": hashlib.sha256(row.text.encode()).hexdigest(),
                     "gold_labels": sorted(row.labels)} for row in selected],
        "label_support": dict(Counter(label for row in selected for label in row.labels)),
        "prior_test_exposure": True,
    })
    contract = freeze_json(output_dir / "selection.json", {
        "version": 1, "split": "dev", "dev_manifest_fingerprint": manifest["fingerprint"],
        "generation_config": generation, "model_digest": model_digest,
        "schema_sha256": fingerprint(GOEMOTIONS_RESPONSE_SCHEMA),
        "prompt_templates": {variant: build_prompt("{comment}", list(LABELS), include_definitions=True,
                                                   structured=True, variant=variant) for variant in PROMPT_VARIANTS},
        "criteria": {"micro_f1": "strictly_higher", "macro_f1": "nondecreasing",
                     "parse_failure_rate_tolerance": 0.0, "paired_bootstrap_samples": 1000,
                     "seed": seed, "correction_95pct_lower_bound": "strictly_positive",
                     "per_label_harm_tolerance": 0.0, "integrated_cost_required": True},
        "classical_trials": [{"features": "word", "balanced": False, "seed": seed}],
        "safe_default": "affect_telemetry_preserve_labels", "correction_policy": "preserve_labels_v1",
        "production_promoted": False, "test_inference_enabled": False,
    })
    return {"manifest": manifest, "selection": contract}


def compare_development(baseline: dict[str, Any], candidate: dict[str, Any],
                        manifest_path: Path, selection_path: Path, *, mode: str) -> dict[str, Any]:
    """Compare separate prompt/training runs or identical-response policies, never pool them."""
    manifest, contract = read_frozen_json(manifest_path), read_frozen_json(selection_path)
    if contract["dev_manifest_fingerprint"] != manifest["fingerprint"] or manifest["split"] != "dev":
        raise ValueError("Selection requires the frozen dev manifest")
    if mode not in {"prompt", "trained_baseline", "same_response"}:
        raise ValueError("Unknown comparison mode")
    expected_ids = [row["id"] for row in manifest["samples"]]
    for bundle in (baseline, candidate):
        if bundle.get("split") != "dev" or bundle.get("dev_manifest_fingerprint") != manifest["fingerprint"]:
            raise ValueError("Comparison is not bound to the selected dev population")
        if not expected_ids or [row["id"] for row in bundle["records"]] != expected_ids:
            raise ValueError("Comparison IDs/order differ from frozen dev manifest")
        for sample, row in zip(manifest["samples"], bundle["records"], strict=True):
            if (sorted(row["gold_labels"]) != sample["gold_labels"]
                    or hashlib.sha256(row["text"].encode()).hexdigest() != sample["text_sha256"]):
                raise ValueError("Record text/gold differs from frozen dev population")
    before, after = baseline["records"], candidate["records"]
    if any(not set(row["predicted_labels"]).issubset(LABELS) for row in before + after):
        raise ValueError("Predicted labels are outside the official taxonomy")
    if mode == "trained_baseline" and any(
        bundle.get("system_family") != "trained_baseline"
        or not isinstance(bundle.get("artifact_fingerprint"), str)
        or len(bundle["artifact_fingerprint"]) != 64 for bundle in (baseline, candidate)
    ):
        raise ValueError("Trained comparison requires frozen trained-baseline identities")
    if any(left["gold_labels"] != right["gold_labels"] for left, right in zip(before, after, strict=True)):
        raise ValueError("Paired gold labels differ")
    for bundle in (baseline, candidate) if mode != "trained_baseline" else ():
        for field in ("generation_config", "model_digest", "schema_sha256"):
            if bundle.get(field) != contract[field]:
                raise ValueError(f"Comparison changed {field}")
        if bundle.get("prompt_variant") not in PROMPT_VARIANTS:
            raise ValueError("Unknown prompt variant")
        for row in bundle["records"]:
            if hashlib.sha256(row["raw_model_response"].encode()).hexdigest() != row["raw_model_response_hash"]:
                raise ValueError("Raw response hash mismatch")
            expected_prompt = contract["prompt_templates"][bundle["prompt_variant"]].replace("{comment}", row["text"])
            if row.get("benchmark_prompt") != expected_prompt:
                raise ValueError("Response is not bound to the exact prompt")
    if mode == "same_response" and any(row.get("parse_error") for row in before + after):
        raise ValueError("Parse failures are excluded from valid same-response correction comparisons")
    if mode == "prompt" and any(not row.get("classification_provider", {}).get("available")
                               or (row["classification_provider"].get("successes", 0)
                                   + row["classification_provider"].get("cache_hits", 0)) < 1
                               for row in before + after):
        raise ValueError("Prompt comparison requires actual request/cache provenance")
    if mode == "prompt" and baseline["prompt_variant"] == candidate["prompt_variant"]:
        raise ValueError("Prompt comparison needs separate prompt identities")
    if mode == "same_response" and any(
        left["raw_model_response_hash"] != right["raw_model_response_hash"]
        or left["benchmark_prompt"] != right["benchmark_prompt"]
        for left, right in zip(before, after, strict=True)
    ):
        raise ValueError("Same-response comparison requires identical response and prompt hashes")
    gold = [frozenset(row["gold_labels"]) for row in before]
    initial = [frozenset(row["predicted_labels"]) for row in before]
    final = [frozenset(row["predicted_labels"]) for row in after]
    initial_metrics, final_metrics = (evaluate(gold, rows, list(LABELS)) for rows in (initial, final))
    criteria = contract["criteria"]
    uncertainty = paired_bootstrap_sample_f1(gold, initial, final, samples=criteria["paired_bootstrap_samples"],
                                             seed=criteria["seed"])
    parse_rates = [sum(bool(row.get("parse_error")) for row in records) / len(records) for records in (before, after)]
    passed = (final_metrics["micro_f1"] > initial_metrics["micro_f1"]
              and final_metrics["macro_f1"] >= initial_metrics["macro_f1"]
              and parse_rates[1] <= parse_rates[0] + criteria["parse_failure_rate_tolerance"])
    populations = {}
    if mode == "same_response":
        for name, indices in {
            "neutral_only": [i for i, row in enumerate(initial) if row == {"neutral"}],
            "neutral_plus_emotion": [i for i, row in enumerate(initial) if "neutral" in row and len(row) > 1],
            "other": [i for i, row in enumerate(initial) if "neutral" not in row],
        }.items():
            populations[name] = {"count": len(indices), "paired_sample_f1": paired_bootstrap_sample_f1(
                [gold[i] for i in indices], [initial[i] for i in indices], [final[i] for i in indices],
                samples=criteria["paired_bootstrap_samples"], seed=criteria["seed"])}
    return {"mode": mode, "population": len(gold), "initial": initial_metrics, "final": final_metrics,
            "paired_sample_f1": uncertainty, "parse_failure_rates": parse_rates,
            "development_metric_gate_passed": passed, "production_promoted": False,
            "correction_admitted": False, "correction_reason": "no_validated_calibrated_policy_or_integrated_cost",
            "populations": populations, "attribution": mode,
            "status": "metric_candidate_pending_cost" if passed else "rejected_or_inconclusive"}
