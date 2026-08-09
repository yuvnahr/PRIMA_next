"""Canonical, crash-safe HotpotQA execution."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import random
import secrets
import shutil
import subprocess  # nosec B404
import threading
import time
from collections import Counter
from collections.abc import Callable
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
from benchmarks.hotpotqa.config import (
    CONTEXT_ALIASES,
    CONTEXT_SOURCES,
    MAX_HOPS,
    MODEL,
    OUTPUT_PATH,
    PARALLEL_WORKERS,
    PROVIDER,
    REASONING_MODE,
    SEED,
    TOP_K,
)
from benchmarks.hotpotqa.console import HotpotQATerminalReporter
from benchmarks.hotpotqa.convert_to_json import convert_validation_set
from benchmarks.hotpotqa.data_sources import DATA_SOURCES, choose_dataset_set, get_data_source
from benchmarks.hotpotqa.evaluate import (
    HotpotQAEvaluator,
    project_supporting_facts,
    score_hotpot_record,
    write_predictions,
)
from benchmarks.hotpotqa.loader import HotpotQADataset
from config.runtime_mode import RuntimeMode
from llm.generation_config import GenerationConfig
from memory.maintenance.background_supervisor import MaintenanceBarrier, MaintenanceMode
from runtime.contracts import (
    DiagnosticMode,
    ExecutionProfile,
    ExecutionStatus,
    PrimaRequest,
    PrimaResponse,
    TaskKind,
)
from runtime.prima_runtime import PrimaRuntime
from runtime.route_profiles import select_route

HOTPOT_PROFILES = (
    ExecutionProfile.MODEL_ONLY, ExecutionProfile.SIMPLE_RAG, ExecutionProfile.PRIMA_FULL,
)
FAILURE_STAGES = {
    "INGESTION_FAILURE": "ingestion", "RUNTIME_FAILURE": "runtime",
    "RETRIEVAL_MISS": "retrieval", "REASONING_STOP_BUDGET": "reasoning",
    "SUPPORTING_FACT_PROJECTION_FAILURE": "supporting_fact",
    "ANSWER_PARSE_FAILURE": "answer", "ANSWER_SCORING_FAILURE": "scoring",
}
_BUDGET_STOPS = {
    "max_hops", "max_retrieval_calls", "max_documents", "max_context_tokens", "time_budget",
}


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        return subprocess.run(  # noqa: S603  # nosec B603
            [git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_dataset(
    dataset_set: str | None, dataset_path: str | Path | None, refresh_data: bool = False,
) -> tuple[str, Path]:
    if dataset_path is not None:
        path = Path(dataset_path)
        if refresh_data:
            if dataset_set is None:
                raise ValueError("--refresh-data requires --dataset-set with --dataset-path")
            path = convert_validation_set(dataset_set, path, force=True)
        elif not path.exists() and dataset_set is not None:
            path = convert_validation_set(dataset_set, path)
        return dataset_set or "custom", path
    if dataset_set is None:
        raise ValueError("Provide --dataset-set or --dataset-path")
    source = get_data_source(dataset_set)
    path = source.default_path
    if refresh_data or not path.exists():
        path = convert_validation_set(dataset_set, path, force=refresh_data)
    return dataset_set, path


def resolve_mode(mode: str | None, dataset_set: str, explicit_path: bool) -> str:
    requested = CONTEXT_ALIASES.get(mode, mode) if mode else None
    expected = None
    if dataset_set in DATA_SOURCES:
        source_mode = get_data_source(dataset_set).mode
        expected = CONTEXT_ALIASES.get(source_mode, source_mode)
    if requested is None:
        if expected is None:
            raise ValueError("--mode is required with a custom --dataset-path")
        return expected
    if requested not in CONTEXT_SOURCES:
        raise ValueError(f"Unsupported HotpotQA context mode: {requested}")
    if not explicit_path and expected is not None and requested not in {expected, "oracle"}:
        raise ValueError(f"Mode {requested!r} is incompatible with dataset set {dataset_set!r}")
    return requested


def select_conversations(
    conversations: list[Any], sampling: str, seed: int | None, offset: int, max_samples: int,
) -> list[Any]:
    selected = list(conversations)
    if sampling == "random":
        if seed is None:
            raise ValueError("Random sampling requires a resolved seed")
        random.Random(seed).shuffle(selected)  # noqa: S311  # nosec B311
    return selected[offset : offset + max_samples if max_samples else None]


class SharedRuntimeFactory:
    """Create isolated runtimes while sharing one heavyweight model client."""

    def __init__(self, factory: Callable[..., Any]) -> None:
        self.factory, self.client, self.lock = factory, None, threading.Lock()

    def create(self, **kwargs: Any) -> Any:
        with self.lock:
            if self.client is not None:
                kwargs["llm_client"] = self.client
            runtime = self.factory(**kwargs)
            if self.client is None:
                self.client = getattr(runtime, "llm_client", None)
            return runtime


async def _execute_case(
    conversation: Any, runtime: Any, profile: ExecutionProfile,
    generation: GenerationConfig, top_k: int, reasoning_mode: str,
    max_hops: int, maintenance: MaintenanceMode,
) -> tuple[PrimaResponse | None, PrimaResponse | None, float, float]:
    ingestion_ms = 0.0
    if hasattr(runtime, "start_maintenance"):
        await runtime.start_maintenance()
    try:
        for turn in conversation.turns:
            started = time.perf_counter()
            response = await runtime.execute(PrimaRequest(
                task_kind=TaskKind.DOCUMENT_INGESTION,
                profile=ExecutionProfile.INGESTION_ONLY,
                input_text=turn.text,
                session_id=conversation.id,
                metadata={"document_metadata": dict(turn.metadata)},
            ))
            ingestion_ms += (time.perf_counter() - started) * 1000
            if response.status is not ExecutionStatus.COMPLETED:
                return response, None, ingestion_ms, 0.0
        if hasattr(runtime, "apply_maintenance_barrier"):
            await runtime.apply_maintenance_barrier(maintenance, MaintenanceBarrier.BEFORE_QUESTION)
        started = time.perf_counter()
        answer = await runtime.execute(PrimaRequest(
            task_kind=TaskKind.FACTUAL_QA,
            profile=profile,
            input_text=conversation.questions[0].question,
            session_id=conversation.id,
            generation_config=generation,
            diagnostic_mode=DiagnosticMode.DIAGNOSTIC,
            metadata={
                "top_k": top_k, "max_hops": max_hops,
                "reasoning_mode": reasoning_mode, "max_context_tokens": 1600,
            },
        ))
        return None, answer, ingestion_ms, (time.perf_counter() - started) * 1000
    finally:
        if hasattr(runtime, "stop_maintenance"):
            await runtime.stop_maintenance(graceful=True)


def _diagnostics(response: PrimaResponse | None) -> dict[str, Any]:
    if response is None:
        return {"hops": 0, "retrievals": 0, "reflections": 0, "evidence_ids": [],
                "model_calls": 0, "model_usage": {}, "stop_reason": "error"}
    return {
        "hops": int(response.output_data.get("hop_count", 0)),
        "retrievals": response.diagnostics.retrieval_count,
        "reflections": response.diagnostics.reflection_count,
        "evidence_ids": [item.source_id for item in response.evidence],
        "model_calls": response.diagnostics.model_call_count,
        "model_usage": dict(response.diagnostics.model_usage),
        "stop_reason": response.output_data.get("stop_reason"),
    }


def _failure(record: dict[str, Any], profile: ExecutionProfile) -> str | None:
    if record["ingestion_error"]:
        return "INGESTION_FAILURE"
    if record["runtime_error"]:
        return "RUNTIME_FAILURE"
    if not record["prediction"].strip():
        return "ANSWER_PARSE_FAILURE"
    if profile is not ExecutionProfile.MODEL_ONLY and not record["evidence_count"]:
        return "RETRIEVAL_MISS"
    if record["final_stop_reason"] in _BUDGET_STOPS:
        return "REASONING_STOP_BUDGET"
    if record["evidence_count"] and not record["supporting_fact_prediction_count"]:
        return "SUPPORTING_FACT_PROJECTION_FAILURE"
    return None


def run_sample(
    conversation: Any, profile: ExecutionProfile, generation: GenerationConfig,
    top_k: int, reasoning_mode: str, max_hops: int,
    runtime_pool: SharedRuntimeFactory, maintenance: MaintenanceMode,
) -> dict[str, Any]:
    started = time.perf_counter()
    runtime = runtime_pool.create(
        mode=RuntimeMode.BENCHMARK, memory_backend="in_memory",
        maintenance_enabled=maintenance is not MaintenanceMode.DISABLED,
        generation_config=generation,
    )
    ingestion_failure, response, ingestion_ms, reasoning_ms = asyncio.run(_execute_case(
        conversation, runtime, profile, generation, top_k, reasoning_mode, max_hops, maintenance,
    ))
    response_dict = response.to_dict() if response is not None else {}
    supporting, provenance = project_supporting_facts(response_dict)
    diagnostic = _diagnostics(response)
    ingestion_error = (
        "; ".join(ingestion_failure.errors) or ingestion_failure.status.value
        if ingestion_failure is not None else None
    )
    runtime_error = None
    if ingestion_failure is None and (response is None or response.status is not ExecutionStatus.COMPLETED):
        runtime_error = "; ".join(response.errors if response else ()) or "runtime returned no response"
    question = conversation.questions[0]
    record = {
        "schema_version": "1.0", "sample_id": conversation.id,
        "runtime_profile": profile.value, "type": question.category,
        "level": conversation.metadata.get("level"), "question": question.question,
        "expected_answer": question.answer,
        "gold_supporting_facts": question.metadata.get("supporting_facts", []),
        "prediction": response.output_text if response and response.output_text else "",
        "supporting_facts": supporting, "supporting_fact_provenance": provenance,
        "raw_runtime_response": response_dict,
        "ingestion_time_ms": round(ingestion_ms, 3),
        "reasoning_time_ms": round(reasoning_ms, 3),
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "ingestion_error": ingestion_error, "runtime_error": runtime_error,
        "reasoning_hops": diagnostic["hops"], "retrieval_calls": diagnostic["retrievals"],
        "reflection_interventions": diagnostic["reflections"],
        "evidence_ids": diagnostic["evidence_ids"],
        "model_call_count": diagnostic["model_calls"],
        "model_usage": diagnostic["model_usage"],
        "final_stop_reason": diagnostic["stop_reason"],
        "evidence_count": len(diagnostic["evidence_ids"]),
        "supporting_fact_prediction_count": len(supporting),
        "scored": False, "execution_failed": bool(ingestion_error or runtime_error),
    }
    record["failure_category"] = _failure(record, profile)
    record["failure_stage"] = FAILURE_STAGES.get(str(record["failure_category"]))
    return record


def _score(records: list[dict[str, Any]]) -> dict[str, float]:
    for record in records:
        try:
            scores = score_hotpot_record(record)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            record.update(
                failure_category="ANSWER_SCORING_FAILURE", failure_stage="scoring",
                runtime_error=f"{type(exc).__name__}: {exc}", execution_failed=True,
                per_item_scores={}, scored=False,
            )
            continue
        record["per_item_scores"], record["scored"] = scores, bool(scores)
        if scores and scores["joint_em"] < 1.0 and record.get("failure_category") is None:
            record.update(failure_category="ANSWER_SCORING_FAILURE", failure_stage="scoring")
    return HotpotQAEvaluator().evaluate(records)


def _failure_record(record: dict[str, Any]) -> FailureRecord | None:
    subcode = record.get("failure_category")
    if not subcode:
        return None
    failed = bool(record["execution_failed"])
    return FailureRecord(
        case_id=record["sample_id"],
        category=FailureCategory.RUNTIME if failed else FailureCategory.EVALUATION,
        subcode=subcode,
        stage=LifecycleStage.EXECUTE_CASE if failed else LifecycleStage.EVALUATE,
        message=record.get("ingestion_error") or record.get("runtime_error") or subcode,
        details={"hotpot_record": record},
    )


def _prediction_record(record: dict[str, Any], started_at: datetime) -> PredictionRecord:
    usage = record.get("model_usage", {})
    return PredictionRecord(
        case_id=record["sample_id"], mode=BenchmarkMode.CONVERSATION_QA,
        prediction=record["prediction"],
        timing=ItemTiming(
            started_at=started_at, finished_at=datetime.now(timezone.utc),
            total_ms=record["total_latency_ms"], provider_ms=record["reasoning_time_ms"],
        ),
        tokens=TokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        ),
        diagnostics={"hotpot_record": record},
    )


def _checkpoint(store: BenchmarkArtifactStore, record: dict[str, Any]) -> None:
    failure = _failure_record(record)
    if record["execution_failed"]:
        if failure is None:
            raise ValueError("failed HotpotQA record requires failure details")
        store.append_failure(failure)
        store.append_checkpoint(CheckpointRecord(
            case_id=record["sample_id"], status=RunStatus.FAILED, failure=failure,
        ))
        return
    prediction = _prediction_record(record, datetime.now(timezone.utc))
    store.append_prediction(prediction)
    if failure is not None:
        store.append_failure(failure)
    store.append_checkpoint(CheckpointRecord(
        case_id=record["sample_id"], status=RunStatus.COMPLETE,
        prediction=prediction, failure=failure,
    ))


def _records(store: BenchmarkArtifactStore) -> list[dict[str, Any]]:
    rows = []
    for checkpoint in store.checkpoints():
        if checkpoint.prediction:
            record = checkpoint.prediction.diagnostics.get("hotpot_record")
        elif checkpoint.failure:
            record = checkpoint.failure.details.get("hotpot_record")
        else:
            record = None
        if isinstance(record, dict):
            rows.append(dict(record))
    return rows


def _dependencies() -> dict[str, str]:
    versions = {}
    for package in ("pydantic", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _manifest(
    dataset: Path, dataset_set: str, mode: str, profile: ExecutionProfile,
    generation: GenerationConfig, selected_ids: list[str], seed: int | None,
    benchmark_config: dict[str, Any],
) -> BenchmarkManifest:
    commit = git_commit()
    root = Path(__file__).parents[2]
    route = select_route(TaskKind.FACTUAL_QA, profile)
    return BenchmarkManifest(
        campaign_id=f"hotpotqa-{uuid4()}",
        benchmark=BenchmarkSpec(
            name="HotpotQA", version="1.0", mode=BenchmarkMode.CONVERSATION_QA,
            dataset_name=dataset_set,
            description="Official HotpotQA metrics over canonical PRIMA profiles.",
        ),
        source_fingerprint=None if commit else fingerprint(Path(__file__)), git_commit=commit,
        dataset_hash=fingerprint(dataset), selected_ids=tuple(selected_ids),
        provider=generation.provider, model=generation.model,
        generation_config=generation.to_dict(), benchmark_config=benchmark_config,
        runtime_profile=profile.value,
        active_capabilities={
            "planned_components": [component.value for component in route.components],
            "context_mode": mode, "context_source": CONTEXT_SOURCES[mode],
            "headline_eligible": mode != "oracle",
        },
        repository_mode="benchmark:in_memory",
        prompt_hashes={
            "answer_generation": fingerprint(root / "workflow" / "answer_generation.py"),
            "prompt_builder": fingerprint(root / "llm" / "prompt_builder.py"),
        },
        seed=seed, dependencies=_dependencies(),
        hardware={
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(),
        },
    )


def _write_artifacts(
    records: list[dict[str, Any]], output_dir: Path,
    metrics: dict[str, float], summary: dict[str, Any],
) -> dict[str, str]:
    raw, metrics_dir = output_dir / "raw", output_dir / "metrics"
    predictions = {
        "answer": {row["sample_id"]: row["prediction"] for row in records},
        "sp": {row["sample_id"]: row["supporting_facts"] for row in records},
    }
    prediction_path = write_predictions(predictions, raw / "hotpot_predictions.json")
    atomic_write_json(raw / "hotpot_failures.json", [r for r in records if r.get("failure_category")])
    atomic_write_json(raw / "retrieval_diagnostics.json", [{
        key: row.get(key) for key in (
            "sample_id", "retrieval_calls", "reasoning_hops", "evidence_ids",
            "supporting_fact_provenance",
        )
    } for row in records])
    atomic_write_json(raw / "reasoning_diagnostics.json", [{
        key: row.get(key) for key in (
            "sample_id", "reflection_interventions", "final_stop_reason",
            "model_call_count", "model_usage", "failure_category", "total_latency_ms",
        )
    } for row in records])
    metrics_path = metrics_dir / "hotpot_metrics.json"
    atomic_write_json(metrics_path, metrics)
    atomic_write_json(metrics_dir / "supporting_fact_metrics.json", {
        key: metrics[key] for key in ("sp_em", "sp_f1", "sp_prec", "sp_recall")
    })
    atomic_write_json(metrics_dir / "hotpot_summary.json", summary | metrics)
    return {
        "predictions": str(prediction_path), "metrics": str(metrics_path),
        "manifest": str(output_dir / "manifest.json"),
        "checkpoint": str(output_dir / "checkpoints" / "records.jsonl"),
        "logs": str(output_dir / "logs"),
    }


def run_hotpotqa_experiment(
    *, mode: str | None = None,
    runtime_profile: ExecutionProfile | str = ExecutionProfile.PRIMA_FULL,
    dataset_path: str | Path | None = None, dataset_set: str | None = None,
    refresh_data: bool = False, output_path: str | Path = OUTPUT_PATH,
    max_samples: int = 0, offset: int = 0, seed: int | None = SEED,
    sampling: str = "random", provider: str = PROVIDER, model: str = MODEL,
    top_k: int = TOP_K, reasoning_mode: str = REASONING_MODE,
    max_hops: int = MAX_HOPS, parallel_workers: int = PARALLEL_WORKERS,
    progress: bool = False, quiet: bool = False, resume: bool = False,
    runtime_factory: Callable[..., Any] = PrimaRuntime,
    maintenance_mode: MaintenanceMode | str = MaintenanceMode.DISABLED,
    _runtime_pool: SharedRuntimeFactory | None = None,
) -> dict[str, Any]:
    if parallel_workers < 1 or offset < 0 or max_samples < 0:
        raise ValueError("Counts must be non-negative and workers must be positive")
    if sampling not in {"random", "sequential"}:
        raise ValueError("sampling must be 'random' or 'sequential'")
    profile = ExecutionProfile(runtime_profile)
    if profile not in HOTPOT_PROFILES:
        raise ValueError("HotpotQA requires model_only, simple_rag, or prima_full")
    maintenance = MaintenanceMode(maintenance_mode)
    explicit_path = dataset_path is not None
    dataset_set, dataset = resolve_dataset(dataset_set, dataset_path, refresh_data)
    mode = resolve_mode(mode, dataset_set, explicit_path)
    output_dir = Path(output_path) / mode / profile.value
    store = BenchmarkArtifactStore(output_dir)
    if not resume and store.layout.manifest.exists():
        raise ValueError(f"Output already contains a campaign; use --resume or a new path: {output_dir}")
    existing_manifest = store.read_manifest() if resume else None
    resolved_seed = seed
    if sampling == "random" and resolved_seed is None:
        resolved_seed = existing_manifest.seed if existing_manifest else secrets.randbits(63)
    conversations = select_conversations(
        list(HotpotQADataset(dataset, mode).conversations()),
        sampling, resolved_seed, offset, max_samples,
    )
    selected_ids = [item.id for item in conversations]
    generation = GenerationConfig(model=model, provider=provider, seed=resolved_seed)
    benchmark_config = {
        "dataset_set": dataset_set, "context_mode": mode,
        "context_source": CONTEXT_SOURCES[mode], "oracle_diagnostic_only": mode == "oracle",
        "sampling": sampling, "offset": offset, "max_samples": max_samples,
        "top_k": top_k, "reasoning_mode": reasoning_mode, "max_hops": max_hops,
        "maintenance_mode": maintenance.value,
    }
    manifest = _manifest(
        dataset, dataset_set, mode, profile, generation,
        selected_ids, resolved_seed, benchmark_config,
    )
    if resume:
        if existing_manifest is None:
            raise ValueError(f"Cannot resume without manifest: {store.layout.manifest}")
        manifest = resume_manifest(existing_manifest, manifest)
    store.initialize(manifest)
    existing = _records(store) if resume else []
    completed_ids = {row["sample_id"] for row in existing}
    pending = [item for item in conversations if item.id not in completed_ids]
    pool = _runtime_pool or SharedRuntimeFactory(runtime_factory)
    reporter = HotpotQATerminalReporter(quiet=quiet, progress=progress)
    report_config = {
        "dataset_set": dataset_set, "mode": mode, "runtime_profile": profile.value,
        "dataset_path": dataset.resolve(), "sample_count": len(conversations),
        "sampling_strategy": sampling, "resolved_seed": resolved_seed,
        "provider": provider, "model": model, "reasoning_mode": reasoning_mode,
        "top_k": top_k, "max_hops": max_hops, "output_dir": output_dir, "resume": resume,
    }
    reporter.header(report_config)
    campaign_started, wall_started = datetime.now(timezone.utc), time.perf_counter()

    def execute(item: Any) -> dict[str, Any]:
        return run_sample(
            item, profile, generation, top_k, reasoning_mode,
            max_hops, pool, maintenance,
        )

    iterator = map(execute, pending)
    executor = None
    if parallel_workers > 1:
        executor = ThreadPoolExecutor(max_workers=parallel_workers)
        iterator = executor.map(execute, pending)
    new_records = []
    try:
        for index, record in enumerate(iterator, 1):
            _score([record])
            _checkpoint(store, record)
            new_records.append(record)
            reporter.result(index, len(pending), record)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    records = existing + new_records
    metrics = _score(records)
    completed = sum(not row["execution_failed"] for row in records)
    failed, scored = len(records) - completed, sum(bool(row["scored"]) for row in records)
    categories = Counter(str(row["failure_category"]) for row in records if row.get("failure_category"))
    summary_data = {
        "schema_version": "1.0", "processed": len(records), "completed": completed,
        "failed": failed, "scored": scored, "failure_total": sum(categories.values()),
        "failure_categories": dict(categories), "context_source": CONTEXT_SOURCES[mode],
        "headline_eligible": mode != "oracle",
    }
    if len(records) != completed + failed:
        raise RuntimeError("HotpotQA completion/failure reconciliation failed")
    artifacts = _write_artifacts(records, output_dir, metrics, summary_data)
    store.finalize(BenchmarkSummary(
        campaign_id=manifest.campaign_id, status=RunStatus.COMPLETE,
        total_cases=len(selected_ids), completed_cases=completed, failed_cases=failed,
        cancelled_cases=0, metrics=summary_data | metrics, started_at=campaign_started,
    ))
    reporter.summary(records, metrics, report_config, artifacts, time.perf_counter() - wall_started)
    return {
        "dataset_set": dataset_set, "mode": mode, "runtime_profile": profile.value,
        "context_source": CONTEXT_SOURCES[mode], "headline_eligible": mode != "oracle",
        "sampling_strategy": sampling, "resolved_seed": resolved_seed,
        **summary_data, "metrics": metrics, "artifacts": artifacts,
    }


def run_paired_hotpotqa_experiment(**kwargs: Any) -> dict[str, dict[str, Any]]:
    """Run canonical profiles over the same selected cases and model settings."""
    kwargs.pop("runtime_profile", None)
    explicit_path = kwargs.get("dataset_path") is not None
    dataset_set, dataset = resolve_dataset(
        kwargs.get("dataset_set"), kwargs.get("dataset_path"), kwargs.get("refresh_data", False)
    )
    mode = resolve_mode(kwargs.get("mode"), dataset_set, explicit_path)
    kwargs.update(
        dataset_set=dataset_set, dataset_path=dataset, mode=mode, refresh_data=False
    )
    resolved_seed = kwargs.get("seed", SEED)
    if kwargs.get("sampling", "random") == "random" and resolved_seed is None:
        if kwargs.get("resume"):
            prior = BenchmarkArtifactStore(
                Path(kwargs.get("output_path", OUTPUT_PATH)) / mode / ExecutionProfile.MODEL_ONLY.value
            ).read_manifest()
            resolved_seed = prior.seed
        else:
            resolved_seed = secrets.randbits(63)
    kwargs["seed"] = resolved_seed
    pool = SharedRuntimeFactory(kwargs.get("runtime_factory", PrimaRuntime))
    results = {profile.value: run_hotpotqa_experiment(
        **kwargs, runtime_profile=profile, _runtime_pool=pool,
    ) for profile in HOTPOT_PROFILES}
    selected = {
        BenchmarkArtifactStore(Path(result["artifacts"]["manifest"]).parent).read_manifest().selected_ids
        for result in results.values()
    }
    if len(selected) != 1:
        raise RuntimeError("Paired HotpotQA profiles selected different sample IDs")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HotpotQA through canonical PRIMA profiles")
    parser.add_argument("--mode", choices=tuple((*CONTEXT_SOURCES, *CONTEXT_ALIASES)))
    parser.add_argument("--profile", choices=tuple(p.value for p in HOTPOT_PROFILES), default="prima_full")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--dataset-set", choices=tuple(DATA_SOURCES))
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sampling", choices=("random", "sequential"), default="random")
    parser.add_argument("--provider", default=PROVIDER)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--reasoning-mode", default=REASONING_MODE)
    parser.add_argument("--max-hops", type=int, default=MAX_HOPS)
    parser.add_argument("--parallel-workers", type=int, default=PARALLEL_WORKERS)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json-summary", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maintenance-mode", choices=tuple(m.value for m in MaintenanceMode), default="disabled")
    args = parser.parse_args()
    if args.dataset_set is None and args.dataset_path is None:
        try:
            args.dataset_set = choose_dataset_set()
        except ValueError as exc:
            parser.error(str(exc))
    values = vars(args)
    json_summary, paired = values.pop("json_summary"), values.pop("paired")
    values["runtime_profile"] = values.pop("profile")
    try:
        result = run_paired_hotpotqa_experiment(**values) if paired else run_hotpotqa_experiment(**values)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if json_summary:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
