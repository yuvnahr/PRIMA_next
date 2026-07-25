"""Terminal-runnable HotpotQA experiment with checkpoint/resume."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit
from benchmarks.common.runtime_adapter import PrimaRuntimeAdapter
from benchmarks.hotpotqa.checkpoint import append_checkpoint, read_checkpoint, validate_resume
from benchmarks.hotpotqa.config import CONTEXT_SOURCES, MAX_HOPS, MODEL, OLLAMA_URL, OUTPUT_PATH, PARALLEL_WORKERS, PROVIDER, REASONING_MODE, SEED, TOP_K
from benchmarks.hotpotqa.evaluate import HotpotQAEvaluator, project_supporting_facts, write_predictions
from benchmarks.hotpotqa.loader import HotpotQADataset
from benchmarks.hotpotqa.runner import HotpotQARunner
from runtime.prima_runtime import PrimaRuntime

def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def safe_url(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.port: host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, ""))

def git_commit() -> str | None:
    try: return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): return None

def manifest_for(dataset_path: Path, mode: str, provider: str, model: str, reasoning_mode: str, top_k: int, max_hops: int, workers: int, seed: int, sample_count: int, resume: bool) -> dict[str, Any]:
    return {
        "benchmark_name": "HotpotQA", "benchmark_mode": mode, "context_source": CONTEXT_SOURCES[mode],
        "dataset_path": str(dataset_path.resolve()), "dataset_fingerprint": fingerprint(dataset_path), "sample_count": sample_count,
        "provider": provider, "model": model, "ollama_url": safe_url(OLLAMA_URL), "reasoning_mode": reasoning_mode,
        "max_hops": max_hops, "top_k": top_k, "parallel_workers": workers, "seed": seed,
        "python_version": sys.version, "platform": platform.platform(), "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "resume_status": bool(resume),
    }

def failure_category(record: dict[str, Any]) -> str | None:
    if record.get("runtime_error"): return "RUNTIME_ERROR"
    if not record.get("evidence_count"): return "NO_EVIDENCE_RETRIEVED"
    reason = str(record.get("final_stop_reason", ""))
    if reason in {"max_hops", "max_retrieval_calls", "max_documents", "max_context_tokens", "time_budget"}: return "REASONING_BUDGET_EXHAUSTED"
    if reason in {"no_progress", "duplicate_query"}: return "NO_PROGRESS"
    if reason == "contradictory": return "CONTRADICTORY_EVIDENCE"
    if record.get("supporting_fact_prediction_count") == 0: return "SUPPORTING_FACT_PROJECTION_FAILURE"
    if not record.get("prediction", "").strip(): return "ANSWER_SYNTHESIS_FAILURE"
    return None

def run_sample(conversation: Any, output_dir: Path, provider: str, model: str, top_k: int, reasoning_mode: str, max_hops: int, runtime_factory: Callable[..., PrimaRuntime]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        agent = PrimaRuntimeAdapter(runtime_factory=runtime_factory, log_path=output_dir / "logs" / "prima_runtime_adapter.log", document_ingestion=True)
        agent.answer_options = {"provider": provider, "model": model, "top_k": top_k, "reasoning_mode": reasoning_mode, "max_hops": max_hops}
        result = HotpotQARunner(output_dir / "logs").run(agent, [conversation])[0]
        response = dict(result.response.metadata)
        supporting, provenance = project_supporting_facts(response)
        state = result.metadata.get("agent_state", {})
        trace = response.get("trace_summary", [])
        record = {
            "sample_id": result.conversation_id, "question": result.prompt, "expected_answer": result.expected_answer,
            "raw_runtime_response": response, "prediction": result.response.text, "supporting_facts": supporting,
            "supporting_fact_provenance": provenance,
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
        question = conversation.questions[0]
        record = {"sample_id": conversation.id, "question": question.question, "expected_answer": question.answer,
            "gold_supporting_facts": question.metadata.get("supporting_facts", []), "raw_runtime_response": {}, "prediction": "", "supporting_facts": [],
            "supporting_fact_provenance": [], "ingestion_time": 0.0, "reasoning_time": 0.0, "total_latency": time.perf_counter() - started,
            "runtime_error": f"{type(exc).__name__}: {exc}", "reasoning_hops": 0, "retrieval_calls": 0, "reflection_interventions": 0,
            "final_stop_reason": "error", "evidence_count": 0, "supporting_fact_prediction_count": 0}
    record["failure_category"] = failure_category(record)
    category = record["failure_category"] or ""
    record["failure_stage"] = ("runtime" if category in {"INGESTION_FAILURE", "RUNTIME_ERROR"} else "retrieval" if "RETRIEVAL" in category or category == "NO_EVIDENCE_RETRIEVED" else "reasoning" if category in {"REASONING_BUDGET_EXHAUSTED", "NO_PROGRESS", "CONTRADICTORY_EVIDENCE"} else "supporting_fact" if category == "SUPPORTING_FACT_PROJECTION_FAILURE" else "answer" if category.startswith("ANSWER_") else None)
    return record

def write_artifacts(records: list[dict[str, Any]], output_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    raw, metrics_dir = output_dir / "raw", output_dir / "metrics"
    raw.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True); (output_dir / "logs").mkdir(parents=True, exist_ok=True)
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
    (metrics_dir / "hotpot_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (metrics_dir / "supporting_fact_metrics.json").write_text(json.dumps({key: metrics[key] for key in ("sp_em", "sp_f1", "sp_prec", "sp_recall")}, indent=2), encoding="utf-8")
    summary = {"completed": len(rows), "failures": len(failures), "context_source": manifest["context_source"], **metrics}
    (metrics_dir / "hotpot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (metrics_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"predictions": str(prediction_path), "metrics": str(metrics_dir / "hotpot_metrics.json"), "manifest": str(metrics_dir / "run_manifest.json")}

def run_hotpotqa_experiment(*, mode: str, dataset_path: str | Path, output_path: str | Path = OUTPUT_PATH, max_samples: int = 0, offset: int = 0, seed: int = SEED, provider: str = PROVIDER, model: str = MODEL, top_k: int = TOP_K, reasoning_mode: str = REASONING_MODE, max_hops: int = MAX_HOPS, parallel_workers: int = PARALLEL_WORKERS, progress: bool = False, resume: bool = False, checkpoint_every: int = 1, runtime_factory: Callable[..., PrimaRuntime] = PrimaRuntime) -> dict[str, Any]:
    if checkpoint_every < 1 or parallel_workers < 1 or offset < 0 or max_samples < 0: raise ValueError("Counts must be non-negative and workers/checkpoint interval must be positive.")
    dataset = Path(dataset_path)
    conversations = list(HotpotQADataset(dataset, mode).conversations())
    random.Random(seed).shuffle(conversations)
    conversations = conversations[offset:offset + max_samples if max_samples else None]
    base = Path(output_path); output_dir = base if base.name == mode else base / mode
    checkpoint = output_dir / "raw" / "hotpot_results.jsonl"; manifest_path = output_dir / "metrics" / "run_manifest.json"
    manifest = manifest_for(dataset, mode, provider, model, reasoning_mode, top_k, max_hops, parallel_workers, seed, len(conversations), resume)
    existing = read_checkpoint(checkpoint) if resume else []
    if resume:
        if not manifest_path.exists(): raise ValueError(f"Cannot resume without manifest: {manifest_path}")
        validate_resume(json.loads(manifest_path.read_text(encoding="utf-8")), manifest)
    elif checkpoint.exists(): checkpoint.write_text("", encoding="utf-8")
    completed = {row["sample_id"] for row in existing}
    pending = [item for item in conversations if item.id not in completed]
    output_dir.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    def execute(item: Any) -> dict[str, Any]: return run_sample(item, output_dir, provider, model, top_k, reasoning_mode, max_hops, runtime_factory)
    iterator = map(execute, pending) if parallel_workers == 1 else ThreadPoolExecutor(max_workers=parallel_workers).map(execute, pending)
    records = list(existing)
    for index, record in enumerate(iterator, 1):
        append_checkpoint(checkpoint, record); records.append(record)
        if progress: print(f"[{index}/{len(pending)}] {record['sample_id']} {record.get('failure_category') or 'OK'}", flush=True)
    artifacts = write_artifacts(records, output_dir, manifest)
    return {"mode": mode, "context_source": CONTEXT_SOURCES[mode], "completed": len({row['sample_id'] for row in records}), "artifacts": artifacts}

def main() -> None:
    parser = argparse.ArgumentParser(description="Run HotpotQA through production PRIMA-NEXT.")
    parser.add_argument("--mode", choices=tuple(CONTEXT_SOURCES), required=True); parser.add_argument("--dataset-path", type=Path, required=True); parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--max-samples", type=int, default=0); parser.add_argument("--offset", type=int, default=0); parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--provider", default=PROVIDER); parser.add_argument("--model", default=MODEL); parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--reasoning-mode", default=REASONING_MODE); parser.add_argument("--max-hops", type=int, default=MAX_HOPS); parser.add_argument("--parallel-workers", type=int, default=PARALLEL_WORKERS)
    parser.add_argument("--progress", action="store_true"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args(); print(json.dumps(run_hotpotqa_experiment(**vars(args)), indent=2))

if __name__ == "__main__": main()
