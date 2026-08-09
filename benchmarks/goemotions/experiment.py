"""Canonical, resumable GoEmotions component benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchmarks.common import (
    BenchmarkArtifactStore,
    BenchmarkManifest,
    BenchmarkMode,
    BenchmarkSpec,
    BenchmarkSummary,
    CheckpointRecord,
    FailureCategory,
    FailureRecord,
    ItemTiming,
    LifecycleStage,
    PredictionRecord,
    RunStatus,
    TokenUsage,
    atomic_write_json,
    resume_manifest,
)
from benchmarks.goemotions.dataset import DEFAULT_DATASET_PATH, GoEmotionsExample, load_examples, load_labels
from benchmarks.goemotions.metrics import (
    evaluate,
    paired_bootstrap_sample_f1,
    paired_outcomes,
    probability_metrics,
    render_figures,
)
from benchmarks.goemotions.prompts import build_prompt
from benchmarks.goemotions.schemas import parse_label_response
from benchmarks.goemotions.systems import (
    AffectTelemetrySystem,
    BoundedAffectDecisionSystem,
    ClassifierSettings,
    EncoderSystem,
    GoEmotionsSystem,
    ModelOnlyZeroShotSystem,
    SchemaConstrainedModelOnlySystem,
)
from benchmarks.goemotions.training.data import validate_splits
from llm.generation_config import GenerationConfig, StructuredOutputMode
from llm.llm_client import LLMClient
from llm.provider import ProviderError
from runtime.contracts import ExecutionProfile, TaskKind
from runtime.route_profiles import select_route

OUTPUT_PATH = Path("evaluation/goemotions")
DEFAULT_MODEL = "qwen3.5:4b"
SYSTEM_ALIASES = {
    "qwen_zero_shot": "model_only_zero_shot",
    "qwen_definitions": "schema_constrained_model_only",
    "qwen_schema": "schema_constrained_model_only",
    "qwen_with_prima_telemetry": "affect_telemetry_preserve_labels",
    "prima_qwen": "bounded_prima_affect_decision",
    "prima_goemotions": "trained_encoder",
    "goemotions_pretrained": "trained_encoder",
    "goemotions_deberta": "trained_encoder",
}
SYSTEMS = (
    "model_only_zero_shot",
    "schema_constrained_model_only",
    "affect_telemetry_preserve_labels",
    "bounded_prima_affect_decision",
    "trained_encoder",
)


def run_goemotions_experiment(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_path: Path = OUTPUT_PATH,
    seed: int = 13,
    batch_size: int = 1,
    max_samples: int = 0,
    device: str = "auto",
    parallel_workers: int = 1,
    provider: str = "ollama",
    model: str = DEFAULT_MODEL,
    system: str = "model_only_zero_shot",
    split: str = "test",
    sample_manifest: Path | None = None,
    write_sample_manifest: Path | None = None,
    thresholds: Path | None = None,
    calibration: Path | None = None,
    progress: bool = True,
    resume: bool = False,
    bootstrap_samples: int = 1000,
    generation_config: GenerationConfig | None = None,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Run a split-safe affect/classification campaign with per-example resume."""

    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test.")
    if batch_size < 1 or parallel_workers < 1 or bootstrap_samples < 1:
        raise ValueError("batch size, workers, and bootstrap samples must be positive.")
    if sample_manifest and max_samples:
        raise ValueError("Use either sample_manifest or max_samples, not both.")
    canonical_name = SYSTEM_ALIASES.get(system, system)
    if canonical_name not in SYSTEMS:
        raise ValueError(f"Unsupported GoEmotions system: {system}")

    requested = Path(dataset_path)
    data_dir = requested.parent
    split_validation = validate_splits(data_dir)
    selected_path = data_dir / f"{split}.tsv"
    labels = load_labels(data_dir / "emotions.txt")
    examples = load_examples(selected_path, data_dir / "emotions.txt")
    if sample_manifest:
        examples = _examples_from_manifest(examples, selected_path, sample_manifest)
    elif max_samples > 0:
        examples = random.Random(seed).sample(examples, min(max_samples, len(examples)))  # noqa: S311  # nosec B311
    if write_sample_manifest:
        _write_sample_manifest(examples, selected_path, split, seed, write_sample_manifest)

    generation = generation_config or GenerationConfig(
        model=_effective_model(canonical_name, model),
        provider=provider,
        temperature=0.0,
        seed=seed,
        max_output_tokens=128,
        structured_output=StructuredOutputMode.NONE
        if canonical_name == "model_only_zero_shot"
        else StructuredOutputMode.JSON_SCHEMA,
    )
    classifier = ClassifierSettings(device, batch_size, thresholds, calibration)
    if llm_client is None:
        _preflight_provider(provider, canonical_name, generation.model)
    active_system = _system(canonical_name, generation, classifier, llm_client)
    root = Path(output_path) / canonical_name
    store = BenchmarkArtifactStore(root)
    if not resume and store.layout.manifest.exists():
        raise ValueError(f"Output already contains a campaign; use resume or a new path: {root}")
    manifest = _manifest(
        selected_path,
        examples,
        generation,
        classifier,
        active_system,
        split_validation,
        split,
        seed,
        max_samples,
        bootstrap_samples,
    )
    if resume:
        manifest = resume_manifest(store.read_manifest(), manifest)
    store.initialize(manifest)
    existing = _records(store)
    completed_ids = {record["id"] for record in existing}
    pending = [example for example in examples if example.example_id not in completed_ids]

    def persist_success(
        example: GoEmotionsExample, raw: str, details: dict[str, Any], started_at: datetime, elapsed: float
    ) -> None:
        record = _classify_response(
            example,
            labels,
            raw,
            details,
            str(details.get("benchmark_prompt") or build_prompt(example.text, labels)),
            elapsed,
        )
        usage = details.get("provider_usage", {}) if isinstance(details.get("provider_usage"), dict) else {}
        prediction = PredictionRecord(
            case_id=example.example_id,
            mode=BenchmarkMode.CLASSIFICATION,
            prediction=tuple(record["predicted_labels"]),
            timing=ItemTiming(started_at=started_at, finished_at=datetime.now(timezone.utc), total_ms=elapsed),
            tokens=TokenUsage(
                prompt_tokens=_usage(usage, "prompt_tokens", "prompt_eval_count"),
                completion_tokens=_usage(usage, "completion_tokens", "eval_count"),
                total_tokens=_token_total(usage),
            ),
            diagnostics={"goemotions_record": record},
        )
        store.append_prediction(prediction)
        store.append_checkpoint(
            CheckpointRecord(case_id=example.example_id, status=RunStatus.COMPLETE, prediction=prediction)
        )

    def execute(example: GoEmotionsExample) -> None:
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        try:
            raw, details = active_system.predict(example.text, labels)
            elapsed = (time.perf_counter() - started) * 1000
            persist_success(example, raw, details, started_at, elapsed)
        except Exception as exc:  # one bad example must not destroy a full test run
            failure_record = {
                "id": example.example_id,
                "text": example.text,
                "gold_labels": sorted(example.labels),
                "execution_failed": True,
                "error": str(exc),
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
            failure = FailureRecord(
                case_id=example.example_id,
                category=FailureCategory.RUNTIME,
                stage=LifecycleStage.EXECUTE_CASE,
                message=str(exc),
                details={"goemotions_record": failure_record},
            )
            store.append_failure(failure)
            store.append_checkpoint(
                CheckpointRecord(case_id=example.example_id, status=RunStatus.FAILED, failure=failure)
            )

    if hasattr(active_system, "predict_many"):
        for offset in range(0, len(pending), batch_size):
            chunk = pending[offset : offset + batch_size]
            started_at = datetime.now(timezone.utc)
            started = time.perf_counter()
            try:
                responses = active_system.predict_many([row.text for row in chunk], labels)
                per_example_ms = (time.perf_counter() - started) * 1000 / max(len(chunk), 1)
                for example, (raw, details) in zip(chunk, responses, strict=True):
                    persist_success(example, raw, details, started_at, per_example_ms)
            except Exception:
                for example in chunk:
                    execute(example)
            _progress_update(len(existing) + min(offset + len(chunk), len(pending)), len(examples), progress)
    elif parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            for count, _ in enumerate(executor.map(execute, pending), start=1):
                _progress_update(len(existing) + count, len(examples), progress)
    else:
        for count, example in enumerate(pending, start=1):
            execute(example)
            _progress_update(len(existing) + count, len(examples), progress)

    all_records = _records(store)
    records = [record for record in all_records if not record.get("execution_failed")]
    failures = [record for record in all_records if record.get("execution_failed")]
    metrics = _metrics(records, labels, seed, bootstrap_samples)
    metrics["failed_example_count"] = len(failures)
    metrics["selected_example_count"] = len(examples)
    metrics["completed_example_count"] = len(records)
    if len(all_records) != len(examples) or len(records) + len(failures) != len(examples):
        raise RuntimeError("GoEmotions completion and failure totals do not reconcile.")
    artifacts = _write_artifacts(root, records, failures, metrics, labels, manifest)
    store.finalize(
        BenchmarkSummary(
            campaign_id=manifest.campaign_id,
            status=RunStatus.COMPLETE,
            total_cases=len(examples),
            completed_cases=len(records),
            failed_cases=len(failures),
            cancelled_cases=0,
            metrics=_round_floats(metrics),
            started_at=manifest.created_at,
        )
    )
    return {
        "run_id": manifest.campaign_id,
        "system": active_system.name,
        "scope": "affect/classification component benchmark",
        "dataset_scope": manifest.benchmark_config["dataset_scope"],
        "split": split,
        "samples": len(records),
        "failed": len(failures),
        "metrics": metrics,
        "artifacts": artifacts,
    }


def _system(
    name: str,
    generation: GenerationConfig,
    classifier: ClassifierSettings,
    llm_client: LLMClient | None = None,
) -> GoEmotionsSystem:
    if name == "model_only_zero_shot":
        system: GoEmotionsSystem = ModelOnlyZeroShotSystem(generation)
    elif name == "schema_constrained_model_only":
        system = SchemaConstrainedModelOnlySystem(generation)
    elif name == "affect_telemetry_preserve_labels":
        system = AffectTelemetrySystem(generation)
    elif name == "bounded_prima_affect_decision":
        system = BoundedAffectDecisionSystem(generation)
    elif name == "trained_encoder":
        system = EncoderSystem(generation, "trained_encoder", "trained_encoder", classifier)
    else:
        raise ValueError(f"Unsupported GoEmotions system: {name}")
    if llm_client is not None:
        system.use_client(llm_client)
    return system


def _classify_response(
    example: GoEmotionsExample,
    labels: list[str],
    raw_response: str,
    metadata: dict[str, Any],
    prompt: str,
    latency_ms: float = 0.0,
) -> dict[str, Any]:
    predicted, parse_error = _parse_labels(raw_response, labels)
    final = metadata.get("final_scored_labels")
    if isinstance(final, list) and all(isinstance(value, str) and value in labels for value in final):
        predicted = frozenset(final)
    return {
        "id": example.example_id,
        "text": example.text,
        "gold_labels": sorted(example.labels),
        "predicted_labels": sorted(predicted),
        "prompt": prompt,
        "raw_response": raw_response,
        **metadata,
        "parse_error": parse_error,
        "latency_ms": latency_ms,
        "execution_failed": False,
    }


def _parse_labels(response: str, labels: list[str]) -> tuple[frozenset[str], str | None]:
    return parse_label_response(response, labels)


def _metrics(
    records: list[dict[str, Any]],
    labels: list[str],
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    gold = [frozenset(record["gold_labels"]) for record in records]
    predicted = [frozenset(record["predicted_labels"]) for record in records]
    metrics = evaluate(gold, predicted, labels)
    metrics["parse_failure_count"] = sum(record.get("parse_error") is not None for record in records)
    metrics["parse_failure_rate"] = metrics["parse_failure_count"] / len(records) if records else 0.0
    metrics["empty_output_count"] = sum(not str(record.get("raw_response", "")).strip() for record in records)
    metrics["empty_output_rate"] = metrics["empty_output_count"] / len(records) if records else 0.0
    latencies = [float(record.get("latency_ms", 0.0)) for record in records]
    usage = [record.get("provider_usage", {}) for record in records]
    total_tokens = sum(_token_total(row) or 0 for row in usage if isinstance(row, dict))
    total_seconds = sum(latencies) / 1000
    metrics["performance"] = {
        "total_latency_ms": sum(latencies),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "throughput_examples_per_second": len(records) / total_seconds if total_seconds else 0.0,
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / total_seconds if total_seconds else 0.0,
    }
    probability_rows = [
        record["prediction"]["probabilities"]
        for record in records
        if record.get("probabilities_available") is True and isinstance(record.get("prediction"), dict)
    ]
    metrics["probability_metrics"] = (
        probability_metrics(gold, probability_rows, labels)
        if len(probability_rows) == len(records)
        else {"available": False, "reason": "This system does not emit complete per-label probabilities."}
    )
    if any("baseline_labels" in record for record in records):
        baseline = [frozenset(record.get("baseline_labels", [])) for record in records]
        parse_errors = [record.get("baseline_parse_error") is not None for record in records]
        baseline_empty = [bool(record.get("baseline_empty_output", False)) for record in records]
        metrics["baseline_parse_failure_count"] = sum(parse_errors)
        metrics["baseline_parse_failure_rate"] = sum(parse_errors) / len(records) if records else 0.0
        metrics["baseline_empty_output_count"] = sum(baseline_empty)
        metrics["baseline_empty_output_rate"] = sum(baseline_empty) / len(records) if records else 0.0
        metrics["paired_outcome_groups"] = paired_outcomes(gold, baseline, predicted, parse_errors)
        if any(record.get("system_family") == "bounded_affect_decision" for record in records):
            comparable = [index for index, failed in enumerate(parse_errors) if not failed]
            comparable_gold = [gold[index] for index in comparable]
            comparable_baseline = [baseline[index] for index in comparable]
            comparable_final = [predicted[index] for index in comparable]
            metrics["headline_affect_comparison"] = {
                "excludes_parse_recovery": True,
                "sample_count": len(comparable),
                "baseline": evaluate(comparable_gold, comparable_baseline, labels) if comparable else {},
                "final": evaluate(comparable_gold, comparable_final, labels) if comparable else {},
                "paired_bootstrap_sample_f1": paired_bootstrap_sample_f1(
                    comparable_gold, comparable_baseline, comparable_final, seed=seed, samples=bootstrap_samples
                ),
            }
    return metrics


def _manifest(
    dataset: Path,
    examples: list[GoEmotionsExample],
    generation: GenerationConfig,
    classifier: ClassifierSettings,
    system: GoEmotionsSystem,
    validation: dict[str, object],
    split: str,
    seed: int,
    max_samples: int,
    bootstrap_samples: int,
) -> BenchmarkManifest:
    commit = _git_commit()
    route = select_route(TaskKind.EMOTION_CLASSIFICATION, ExecutionProfile.AFFECT_ONLY)
    counts = validation.get("counts")
    split_count = counts.get(split) if isinstance(counts, dict) else None
    return BenchmarkManifest(
        campaign_id=f"goemotions-{uuid4()}",
        benchmark=BenchmarkSpec(
            name="GoEmotions",
            version="1.0",
            mode=BenchmarkMode.CLASSIFICATION,
            dataset_name=dataset.name,
            description="Affect/classification component benchmark; not a full QA or architecture benchmark.",
        ),
        source_fingerprint=None if commit else _sha256(Path(__file__)),
        git_commit=commit,
        dataset_hash=_sha256(dataset),
        selected_ids=tuple(row.example_id for row in examples),
        provider=generation.provider,
        model=generation.model,
        generation_config=generation.to_dict(),
        benchmark_config={
            "system": system.name,
            "system_family": system.system_family,
            "split": split,
            "dataset_scope": "full_split" if split_count == len(examples) else "sample",
            "max_samples": max_samples,
            "bootstrap_samples": bootstrap_samples,
            "split_validation": validation,
            "classifier": {
                "device": classifier.device,
                "batch_size": classifier.batch_size,
                "thresholds_path": str(classifier.thresholds_path) if classifier.thresholds_path else None,
                "calibration_path": str(classifier.calibration_path) if classifier.calibration_path else None,
            },
        },
        runtime_profile=ExecutionProfile.AFFECT_ONLY.value,
        active_capabilities={
            "task_kind": TaskKind.EMOTION_CLASSIFICATION.value,
            "planned_components": [item.value for item in route.components],
            "scope": "affect/classification only",
        },
        repository_mode="benchmark:in_memory",
        prompt_hashes={"goemotions_prompt": _sha256(Path(__file__).with_name("prompts.py"))},
        seed=seed,
        dependencies=_dependencies(),
        hardware={"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
    )


def _records(store: BenchmarkArtifactStore) -> list[dict[str, Any]]:
    records = []
    for checkpoint in store.checkpoints():
        source = (
            checkpoint.prediction.diagnostics
            if checkpoint.prediction
            else checkpoint.failure.details
            if checkpoint.failure
            else {}
        )
        record = source.get("goemotions_record")
        if isinstance(record, dict):
            records.append(dict(record))
    return records


def _write_artifacts(
    root: Path,
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    metrics: dict[str, Any],
    labels: list[str],
    manifest: BenchmarkManifest,
) -> dict[str, str]:
    paths = _artifact_paths(root)
    for parent in {path.parent for path in paths.values()}:
        parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths["raw"], {"responses": records, "failures": failures})
    atomic_write_json(
        paths["parsed"], [{key: record[key] for key in ("id", "predicted_labels", "parse_error")} for record in records]
    )
    atomic_write_json(
        paths["answers"],
        [{key: record[key] for key in ("id", "gold_labels", "predicted_labels")} for record in records],
    )
    atomic_write_json(paths["prompts"], [{key: record[key] for key in ("id", "prompt")} for record in records])
    atomic_write_json(paths["metrics"], _round_floats(metrics))
    atomic_write_json(
        paths["metadata"],
        {
            "run_id": manifest.campaign_id,
            "scope": "affect/classification component benchmark",
            "manifest_fingerprint": manifest.manifest_fingerprint,
        },
    )
    render_figures(metrics, labels, str(paths["cooccurrence"]), str(paths["labels"]), str(paths["predictions"]))
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(_report(manifest, metrics, len(failures)), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()} | {
        "manifest": str(root / "manifest.json"),
        "checkpoint": str(root / "checkpoints" / "records.jsonl"),
        "summary": str(root / "summary.json"),
    }


def _report(manifest: BenchmarkManifest, metrics: dict[str, Any], failed: int) -> str:
    paired = metrics.get("paired_outcome_groups", {})
    validity = paired.get("baseline_validity", {})
    outcomes = paired.get("affect_outcomes", {})
    return f"""# GoEmotions Affect/Classification Report

This run evaluates affect and multilabel classification behavior through `PrimaRuntime.execute()` using `TaskKind.EMOTION_CLASSIFICATION` and the bounded `affect_only` profile. It does not evaluate retrieval, QA reasoning, tools, world simulation, or the full PRIMA architecture.

- Run ID: `{manifest.campaign_id}`
- System: `{manifest.benchmark_config["system"]}`
- Exact-set accuracy: `{metrics["exact_set_accuracy"]:.6f}`
- Sample F1: `{metrics["sample_f1"]:.6f}`
- Micro F1: `{metrics["micro_f1"]:.6f}`
- Macro F1: `{metrics["macro_f1"]:.6f}`
- Failed examples: `{failed}`
- Baseline parse failures: `{validity.get("baseline_parse_failures", 0)}`
- Parse recoveries (reported separately): `{outcomes.get("prima_parse_recovery", 0)}`
- Label corrections on valid baseline parses: `{outcomes.get("prima_label_correction", 0)}`
- Regressions on valid baseline parses: `{outcomes.get("prima_regression", 0)}`

The headline paired affect comparison excludes baseline parse failures, so parse recovery is never counted invisibly as affect improvement. Trained encoder systems are reported as separate trained baselines, not wrapper gains. The multilabel matrix artifact is label co-occurrence, not a confusion matrix.
"""


def _artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "raw": root / "raw" / "responses.json",
        "parsed": root / "parsed" / "answers.json",
        "answers": root / "answers" / "answers.json",
        "prompts": root / "prompts" / "prompts.json",
        "metrics": root / "metrics" / "metrics.json",
        "report": root / "reports" / "goemotions_report.md",
        "metadata": root / "logs" / "run_metadata.json",
        "cooccurrence": root / "figures" / "label_cooccurrence.png",
        "labels": root / "figures" / "class_distribution.png",
        "predictions": root / "figures" / "prediction_distribution.png",
    }


def _examples_from_manifest(
    examples: list[GoEmotionsExample], dataset_path: Path, manifest_path: Path
) -> list[GoEmotionsExample]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("dataset_sha256") != _sha256(dataset_path):
        raise ValueError("Sample manifest belongs to a different dataset split.")
    by_id = {example.example_id: example for example in examples}
    ids = [
        item["id"] for item in payload.get("samples", []) if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if not ids or len(ids) != len(payload.get("samples", [])) or len(set(ids)) != len(ids):
        raise ValueError("Sample manifest must contain unique sample IDs.")
    try:
        return [by_id[example_id] for example_id in ids]
    except KeyError as exc:
        raise ValueError(f"Sample manifest ID is not in this split: {exc.args[0]}") from exc


def _write_sample_manifest(
    examples: list[GoEmotionsExample], dataset_path: Path, split: str, seed: int, path: Path
) -> None:
    payload: dict[str, Any] = {
        "seed": seed,
        "split": split,
        "dataset_sha256": _sha256(dataset_path),
        "samples": [{"id": row.example_id, "labels": sorted(row.labels)} for row in examples],
    }
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    atomic_write_json(path, payload)


def _preflight_provider(provider: str, system: str, model: str) -> None:
    if provider.lower() != "ollama" or system == "trained_encoder":
        return
    base = "http://localhost:11434"
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as response:  # noqa: S310  # nosec B310
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError):
        raise ProviderError(f"Ollama is not reachable at {base}. Start it with: ollama serve") from None
    if model not in {item.get("name") for item in payload.get("models", []) if isinstance(item, dict)}:
        raise ProviderError(f"Ollama model '{model}' is not installed. Run: ollama pull {model}")


def _effective_model(system: str, model: str) -> str:
    return "SamLowe/roberta-base-go_emotions" if system == "trained_encoder" and model == DEFAULT_MODEL else model


def _usage(value: dict[str, Any], *names: str) -> int | None:
    for name in names:
        if value.get(name) is not None:
            return int(value[name])
    return None


def _token_total(value: dict[str, Any]) -> int | None:
    total = _usage(value, "total_tokens")
    if total is not None:
        return total
    prompt = _usage(value, "prompt_tokens", "prompt_eval_count")
    completion = _usage(value, "completion_tokens", "eval_count")
    return None if prompt is None and completion is None else (prompt or 0) + (completion or 0)


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dependencies() -> dict[str, str]:
    result = {}
    for package in ("pydantic", "numpy", "scikit-learn", "transformers"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def _git_commit() -> str | None:
    try:
        head = Path(".git/HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head or None
        ref = Path(".git") / head.removeprefix("ref: ")
        return ref.read_text(encoding="utf-8").strip() if ref.is_file() else None
    except OSError:
        return None


def _progress_update(current: int, total: int, enabled: bool) -> None:
    if enabled:
        print(f"GoEmotions: {current}/{total} [{current * 100 // max(total, 1)}%]", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded GoEmotions affect/classification benchmark.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system", choices=SYSTEMS + tuple(SYSTEM_ALIASES), default="model_only_zero_shot")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--write-sample-manifest", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    values["progress"] = not values.pop("no_progress")
    try:
        result = run_goemotions_experiment(**values)
    except ProviderError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(_round_floats(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
