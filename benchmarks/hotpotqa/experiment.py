"""Terminal-runnable HotpotQA experiment with checkpoint/resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import secrets
import shutil
import subprocess  # nosec B404
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from benchmarks.common.runtime_adapter import PrimaRuntimeAdapter
from benchmarks.hotpotqa.checkpoint import append_checkpoint, read_checkpoint, validate_resume
from benchmarks.hotpotqa.config import (
    CONTEXT_SOURCES,
    MAX_HOPS,
    MODEL,
    OLLAMA_URL,
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
    write_predictions,
)
from benchmarks.hotpotqa.loader import HotpotQADataset
from benchmarks.hotpotqa.runner import HotpotQARunner
from runtime.prima_runtime import PrimaRuntime


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def safe_url(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, ""))

def git_commit() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        return subprocess.run([git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()  # noqa: S603  # nosec B603
    except (OSError, subprocess.CalledProcessError):
        return None

def resolve_dataset(dataset_set: str | None, dataset_path: str | Path | None, refresh_data: bool = False) -> tuple[str, Path]:
    if dataset_path is not None:
        path = Path(dataset_path)
        resolved_set = dataset_set or "custom"
        if refresh_data:
            if dataset_set is None:
                raise ValueError("--refresh-data requires --dataset-set when --dataset-path is explicit")
            path = convert_validation_set(dataset_set, path, force=True)
        elif not path.exists() and dataset_set is not None:
            path = convert_validation_set(dataset_set, path)
        return resolved_set, path
    if dataset_set is None:
        raise ValueError("Provide --dataset-set or --dataset-path")
    source = get_data_source(dataset_set)
    path = source.default_path
    if refresh_data or not path.exists():
        path = convert_validation_set(dataset_set, path, force=refresh_data)
    return dataset_set, path

def resolve_mode(mode: str | None, dataset_set: str, explicit_path: bool) -> str:
    if mode is None:
        if dataset_set not in DATA_SOURCES:
            raise ValueError("--mode is required when using a custom --dataset-path")
        return get_data_source(dataset_set).mode
    if not explicit_path and dataset_set in DATA_SOURCES and mode not in {get_data_source(dataset_set).mode, "oracle"}:
        raise ValueError(f"Mode {mode!r} is incompatible with dataset set {dataset_set!r}")
    return mode

def select_conversations(conversations: list[Any], sampling: str, seed: int | None, offset: int, max_samples: int) -> list[Any]:
    selected = list(conversations)
    if sampling == "random":
        if seed is None:
            raise ValueError("Random sampling requires a resolved seed")
        random.Random(seed).shuffle(selected)  # noqa: S311  # nosec B311
    return selected[offset:offset + max_samples if max_samples else None]

def manifest_for(dataset_path: Path, dataset_set: str, mode: str, provider: str, model: str, reasoning_mode: str, top_k: int, max_hops: int, workers: int, configured_seed: int | None, resolved_seed: int | None, sampling: str, offset: int, max_samples: int, sample_ids: list[str], resume: bool) -> dict[str, Any]:
    return {
        "benchmark_name": "HotpotQA", "benchmark_mode": mode, "context_source": CONTEXT_SOURCES[mode],
        "dataset_set": dataset_set, "dataset_path": str(dataset_path.resolve()), "dataset_fingerprint": fingerprint(dataset_path),
        "sample_count": len(sample_ids), "selected_sample_ids": sample_ids, "sampling_strategy": sampling,
        "offset": offset, "max_samples": max_samples, "configured_seed": configured_seed, "resolved_seed": resolved_seed,
        "seed": resolved_seed, "provider": provider, "model": model, "ollama_url": safe_url(OLLAMA_URL),
        "reasoning_mode": reasoning_mode, "max_hops": max_hops, "top_k": top_k, "parallel_workers": workers,
        "python_version": sys.version, "platform": platform.platform(), "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "resume_status": bool(resume),
    }

def failure_category(record: dict[str, Any]) -> str | None:
    if record.get("runtime_error"):
        return "RUNTIME_ERROR"
    if not record.get("evidence_count"):
        return "NO_EVIDENCE_RETRIEVED"
    reason = str(record.get("final_stop_reason", ""))
    if reason in {"max_hops", "max_retrieval_calls", "max_documents", "max_context_tokens", "time_budget"}:
        return "REASONING_BUDGET_EXHAUSTED"
    if reason in {"no_progress", "duplicate_query"}:
        return "NO_PROGRESS"
    if reason == "contradictory":
        return "CONTRADICTORY_EVIDENCE"
    if record.get("supporting_fact_prediction_count") == 0:
        return "SUPPORTING_FACT_PROJECTION_FAILURE"
    if not record.get("prediction", "").strip():
        return "ANSWER_SYNTHESIS_FAILURE"
    return None

def run_sample(conversation: Any, output_dir: Path, provider: str, model: str, top_k: int, reasoning_mode: str, max_hops: int, runtime_factory: Callable[..., PrimaRuntime]) -> dict[str, Any]:
    started = time.perf_counter()
    question = conversation.questions[0]
    try:
        agent = PrimaRuntimeAdapter(runtime_factory=runtime_factory, log_path=output_dir / "logs" / "prima_runtime_adapter.log", document_ingestion=True)
        agent.answer_options = {"provider": provider, "model": model, "top_k": top_k, "reasoning_mode": reasoning_mode, "max_hops": max_hops}
        result = HotpotQARunner(output_dir / "logs").run(agent, [conversation])[0]
        response = dict(result.response.metadata)
        supporting, provenance = project_supporting_facts(response)
        state = result.metadata.get("agent_state", {})
        trace = response.get("trace_summary", [])
        record = {
            "sample_id": result.conversation_id, "type": question.category, "level": conversation.metadata.get("level"),
            "question": result.prompt, "expected_answer": result.expected_answer, "raw_runtime_response": response,
            "prediction": result.response.text, "supporting_facts": supporting, "supporting_fact_provenance": provenance,
            "gold_supporting_facts": result.metadata.get("question_metadata", {}).get("supporting_facts", []),
            "ingestion_time": state.get("ingestion_time", 0.0), "reasoning_time": state.get("reasoning_time", 0.0),
            "total_latency": time.perf_counter() - started, "runtime_error": None,
            "reasoning_hops": response.get("hop_count", 0),
            "retrieval_calls": sum(1 for item in trace if item.get("type") == "RetrievalStarted") or response.get("hop_count", 0),
            "reflection_interventions": sum(1 for item in trace if item.get("type") == "ReflectionApplied"),
            "final_stop_reason": response.get("stop_reason"), "evidence_count": len(response.get("evidence_references", [])),
            "supporting_fact_prediction_count": len(supporting),
        }
    except Exception as exc:
        record = {
            "sample_id": conversation.id, "type": question.category, "level": conversation.metadata.get("level"),
            "question": question.question, "expected_answer": question.answer,
            "gold_supporting_facts": question.metadata.get("supporting_facts", []), "raw_runtime_response": {},
            "prediction": "", "supporting_facts": [], "supporting_fact_provenance": [], "ingestion_time": 0.0,
            "reasoning_time": 0.0, "total_latency": time.perf_counter() - started,
            "runtime_error": f"{type(exc).__name__}: {exc}", "reasoning_hops": 0, "retrieval_calls": 0,
            "reflection_interventions": 0, "final_stop_reason": "error", "evidence_count": 0,
            "supporting_fact_prediction_count": 0,
        }
    record["failure_category"] = failure_category(record)
    category = record["failure_category"] or ""
    record["failure_stage"] = ("runtime" if category in {"INGESTION_FAILURE", "RUNTIME_ERROR"} else "retrieval" if "RETRIEVAL" in category or category == "NO_EVIDENCE_RETRIEVED" else "reasoning" if category in {"REASONING_BUDGET_EXHAUSTED", "NO_PROGRESS", "CONTRADICTORY_EVIDENCE"} else "supporting_fact" if category == "SUPPORTING_FACT_PROJECTION_FAILURE" else "answer" if category.startswith("ANSWER_") else None)
    return record

def write_artifacts(records: list[dict[str, Any]], output_dir: Path, manifest: dict[str, Any]) -> tuple[dict[str, str], dict[str, float]]:
    raw, metrics_dir, logs = output_dir / "raw", output_dir / "metrics", output_dir / "logs"
    raw.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    latest = {record["sample_id"]: record for record in records}
    rows = list(latest.values())
    predictions = {"answer": {row["sample_id"]: row["prediction"] for row in rows}, "sp": {row["sample_id"]: row["supporting_facts"] for row in rows}}
    prediction_path = write_predictions(predictions, raw / "hotpot_predictions.json")
    failures = [row for row in rows if row.get("failure_category")]
    (raw / "hotpot_failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    retrieval = [{key: row.get(key) for key in ("sample_id", "retrieval_calls", "evidence_count", "supporting_fact_prediction_count", "supporting_fact_provenance")} for row in rows]
    reasoning = [{key: row.get(key) for key in ("sample_id", "reasoning_hops", "reflection_interventions", "final_stop_reason", "runtime_error", "failure_category", "ingestion_time", "reasoning_time", "total_latency")} for row in rows]
    (raw / "retrieval_diagnostics.json").write_text(json.dumps(retrieval, indent=2), encoding="utf-8")
    (raw / "reasoning_diagnostics.json").write_text(json.dumps(reasoning, indent=2), encoding="utf-8")
    metrics = HotpotQAEvaluator().evaluate(rows)
    metrics_path = metrics_dir / "hotpot_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (metrics_dir / "supporting_fact_metrics.json").write_text(json.dumps({key: metrics[key] for key in ("sp_em", "sp_f1", "sp_prec", "sp_recall")}, indent=2), encoding="utf-8")
    summary = {"completed": len(rows), "failures": len(failures), "context_source": manifest["context_source"], **metrics}
    (metrics_dir / "hotpot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path = metrics_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    artifacts = {"predictions": str(prediction_path), "metrics": str(metrics_path), "manifest": str(manifest_path), "checkpoint": str(raw / "hotpot_results.jsonl"), "logs": str(logs)}
    return artifacts, metrics

def run_hotpotqa_experiment(*, mode: str | None = None, dataset_path: str | Path | None = None, dataset_set: str | None = None, refresh_data: bool = False, output_path: str | Path = OUTPUT_PATH, max_samples: int = 0, offset: int = 0, seed: int | None = SEED, sampling: str = "random", provider: str = PROVIDER, model: str = MODEL, top_k: int = TOP_K, reasoning_mode: str = REASONING_MODE, max_hops: int = MAX_HOPS, parallel_workers: int = PARALLEL_WORKERS, progress: bool = False, quiet: bool = False, resume: bool = False, checkpoint_every: int = 1, runtime_factory: Callable[..., PrimaRuntime] = PrimaRuntime) -> dict[str, Any]:
    if checkpoint_every < 1 or parallel_workers < 1 or offset < 0 or max_samples < 0:
        raise ValueError("Counts must be non-negative and workers/checkpoint interval must be positive.")
    if sampling not in {"random", "sequential"}:
        raise ValueError("sampling must be 'random' or 'sequential'")
    explicit_path = dataset_path is not None
    dataset_set, dataset = resolve_dataset(dataset_set, dataset_path, refresh_data)
    mode = resolve_mode(mode, dataset_set, explicit_path)
    base = Path(output_path); output_dir = base if base.name == mode else base / mode
    checkpoint = output_dir / "raw" / "hotpot_results.jsonl"; manifest_path = output_dir / "metrics" / "run_manifest.json"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if resume and manifest_path.exists() else None
    if resume and existing_manifest is None:
        raise ValueError(f"Cannot resume without manifest: {manifest_path}")
    configured_seed = seed
    resolved_seed = seed
    if sampling == "random" and resolved_seed is None:
        resolved_seed = existing_manifest.get("resolved_seed") if existing_manifest else secrets.randbits(63)
    conversations = select_conversations(list(HotpotQADataset(dataset, mode).conversations()), sampling, resolved_seed, offset, max_samples)
    sample_ids = [item.id for item in conversations]
    manifest = manifest_for(dataset, dataset_set, mode, provider, model, reasoning_mode, top_k, max_hops, parallel_workers, configured_seed, resolved_seed, sampling, offset, max_samples, sample_ids, resume)
    existing = read_checkpoint(checkpoint) if resume else []
    if resume:
        validate_resume(existing_manifest, manifest)
    elif checkpoint.exists():
        checkpoint.write_text("", encoding="utf-8")
    completed = {row["sample_id"] for row in existing}
    pending = [item for item in conversations if item.id not in completed]
    output_dir.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    reporter = HotpotQATerminalReporter(quiet=quiet, progress=progress)
    report_config = {"dataset_set": dataset_set, "mode": mode, "dataset_path": dataset.resolve(), "sample_count": len(conversations), "sampling_strategy": sampling, "resolved_seed": resolved_seed, "provider": provider, "model": model, "reasoning_mode": reasoning_mode, "top_k": top_k, "max_hops": max_hops, "output_dir": output_dir, "resume": resume}
    reporter.header(report_config)
    started = time.perf_counter()
    def execute(item: Any) -> dict[str, Any]:
        return run_sample(item, output_dir, provider, model, top_k, reasoning_mode, max_hops, runtime_factory)
    records = list(existing)
    if parallel_workers == 1:
        iterator = map(execute, pending)
        for index, record in enumerate(iterator, 1):
            append_checkpoint(checkpoint, record); records.append(record); reporter.result(index, len(pending), record)
    else:
        with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
            for index, record in enumerate(pool.map(execute, pending), 1):
                append_checkpoint(checkpoint, record); records.append(record); reporter.result(index, len(pending), record)
    artifacts, metrics = write_artifacts(records, output_dir, manifest)
    elapsed = time.perf_counter() - started
    reporter.summary(list({row["sample_id"]: row for row in records}.values()), metrics, report_config, artifacts, elapsed)
    return {"dataset_set": dataset_set, "mode": mode, "context_source": CONTEXT_SOURCES[mode], "sampling_strategy": sampling, "resolved_seed": resolved_seed, "completed": len({row['sample_id'] for row in records}), "metrics": metrics, "artifacts": artifacts}

def main() -> None:
    parser = argparse.ArgumentParser(description="Run HotpotQA through production PRIMA-NEXT.")
    parser.add_argument("--mode", choices=tuple(CONTEXT_SOURCES)); parser.add_argument("--dataset-set", choices=tuple(DATA_SOURCES)); parser.add_argument("--dataset-path", type=Path); parser.add_argument("--refresh-data", action="store_true"); parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--max-samples", type=int, default=0); parser.add_argument("--offset", type=int, default=0); parser.add_argument("--seed", type=int, default=SEED); parser.add_argument("--sampling", choices=("random", "sequential"), default="random")
    parser.add_argument("--provider", default=PROVIDER); parser.add_argument("--model", default=MODEL); parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--reasoning-mode", default=REASONING_MODE); parser.add_argument("--max-hops", type=int, default=MAX_HOPS); parser.add_argument("--parallel-workers", type=int, default=PARALLEL_WORKERS)
    parser.add_argument("--progress", action="store_true"); parser.add_argument("--quiet", action="store_true"); parser.add_argument("--json-summary", action="store_true"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args()
    if args.dataset_set is None and args.dataset_path is None:
        try:
            args.dataset_set = choose_dataset_set()
        except ValueError as exc:
            parser.error(str(exc))
    json_summary = args.json_summary; del args.json_summary
    try:
        result = run_hotpotqa_experiment(**vars(args))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if json_summary:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
