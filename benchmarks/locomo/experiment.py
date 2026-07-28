"""Configurable LoCoMo benchmark experiment runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmarks.common.interfaces import BenchmarkResult
from benchmarks.common.runtime_adapter import PrimaRuntimeAdapter
from benchmarks.common.utils import configure_benchmark_logger
from benchmarks.locomo.config import (
    LOG_LEVEL,
    MAX_CONVERSATIONS,
    MAX_QUESTIONS,
    MODEL,
    OUTPUT_PATH,
    PARALLEL_WORKERS,
    PROVIDER,
    SEED,
    TOP_K,
)
from benchmarks.locomo.evaluate import LoCoMoEvaluator, write_locomo_artifacts
from benchmarks.locomo.loader import LoCoMoDataset
from benchmarks.locomo.runner import LoCoMoRunner
from memory.embedding_pipeline import current_embedding_metadata
from memory.retrieval.hybrid_fusion import HybridFusionConfig


def run_locomo_experiment(
    max_conversations: int | None = None,
    max_questions: int | None = None,
    seed: int = SEED,
    top_k: int = TOP_K,
    provider: str = PROVIDER,
    model: str = MODEL,
    parallel_workers: int = PARALLEL_WORKERS,
    dataset_path: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run LoCoMo at a configurable scale and write standard artifacts."""

    dataset = LoCoMoDataset(Path(dataset_path) if dataset_path else None) if dataset_path else LoCoMoDataset()
    artifact_root = Path(output_path) if output_path else OUTPUT_PATH
    log_path = artifact_root / "logs"
    configure_benchmark_logger("benchmarks.locomo.loader", log_path, LOG_LEVEL)
    conversations = list(dataset.conversations())
    random.Random(seed).shuffle(conversations)  # noqa: S311  # nosec B311
    max_conversations = MAX_CONVERSATIONS if max_conversations is None else max_conversations
    max_questions = MAX_QUESTIONS if max_questions is None else max_questions
    if max_conversations > 0:
        conversations = conversations[:max_conversations]
    if max_questions > 0:
        conversations = [replace(conversation, questions=conversation.questions[:max_questions]) for conversation in conversations]

    total_questions = sum(len(conversation.questions) for conversation in conversations)
    if parallel_workers > 1 and len(conversations) > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            batches = list(
                executor.map(
                    lambda conversation: run_single_conversation(conversation, top_k, provider, model, log_path),
                    conversations,
                )
            )
        results = [result for batch in batches for result in batch]
    else:
        results = []
        completed = 0
        started = time.monotonic()
        if total_questions:
            print(f"Benchmark progress: 0/{total_questions} questions | ETA: calculating...", flush=True)

        def report_progress(_: BenchmarkResult) -> None:
            nonlocal completed
            completed += 1
            elapsed = time.monotonic() - started
            eta = elapsed / completed * (total_questions - completed)
            width = 24
            filled = round(width * completed / total_questions)
            bar = f"{'#' * filled}{'-' * (width - filled)}"
            ending = "\n" if completed == total_questions else ""
            print(
                f"\rBenchmark [{bar}] {completed}/{total_questions} questions "
                f"({completed / total_questions:.1%}) | elapsed {_duration(elapsed)} | ETA {_duration(eta)}",
                end=ending,
                flush=True,
            )

        for conversation in conversations:
            results.extend(run_single_conversation(conversation, top_k, provider, model, log_path, report_progress))

    metrics = LoCoMoEvaluator().evaluate(results)
    runtime_errors = sum(
        len(item.response.metadata.get("answer_diagnostics", {}).get("errors", ()))
        for item in results
    )
    embedding = current_embedding_metadata()
    paths = write_locomo_artifacts(
        results,
        metrics,
        provider=provider,
        model=model,
        runtime_errors=runtime_errors,
        output_path=artifact_root,
        metadata={
            "dataset_path": str(dataset.dataset_path),
            "dataset_sha256": hashlib.sha256(dataset.dataset_path.read_bytes()).hexdigest(),
            "backend_fingerprint": embedding["backend_fingerprint"],
            "embedding": embedding,
            "retrieval": {
                "top_k": top_k,
                "candidate_pool_size": int(os.getenv("PRIMA_RETRIEVAL_CANDIDATE_POOL_SIZE", "30")),
                "query_expansion_enabled": True,
                "fusion_weights": HybridFusionConfig.from_file().weights,
            },
            "seed": seed,
            "benchmark_version": "locomo10",
            "max_conversations": max_conversations,
            "max_questions": max_questions,
        },
    )
    if runtime_errors:
        raise RuntimeError(f"LoCoMo QA run invalid: {runtime_errors} generation errors; see {paths['raw']}")
    return {
        "conversations": len(conversations),
        "questions": len(results),
        "metrics": metrics,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def run_single_conversation(
    conversation: Any,
    top_k: int,
    provider: str,
    model: str,
    log_path: Path,
    on_question_completed: Callable[[BenchmarkResult], None] | None = None,
) -> list[Any]:
    """Run one conversation with one isolated runtime instance."""

    agent = PrimaRuntimeAdapter(log_path=log_path / "prima_runtime_adapter.log")
    agent.answer_options = {"top_k": top_k, "provider": provider, "model": model}
    return LoCoMoRunner(log_path).run(agent, [conversation], on_question_completed)


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(max(0, round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def main() -> None:
    """Command-line entry point for progressive LoCoMo runs."""

    parser = argparse.ArgumentParser(description="Run a configurable LoCoMo benchmark experiment.")
    parser.add_argument("--max-conversations", type=int, default=MAX_CONVERSATIONS)
    parser.add_argument("--max-questions", type=int, default=MAX_QUESTIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--provider", default=PROVIDER)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--parallel-workers", type=int, default=PARALLEL_WORKERS)
    parser.add_argument("--dataset-path")
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    if args.clean_output and output_path.exists():
        shutil.rmtree(output_path)
    (output_path / "logs").mkdir(parents=True, exist_ok=True)
    result = run_locomo_experiment(
        max_conversations=args.max_conversations,
        max_questions=args.max_questions,
        seed=args.seed,
        top_k=args.top_k,
        provider=args.provider,
        model=args.model,
        parallel_workers=args.parallel_workers,
        dataset_path=args.dataset_path,
        output_path=str(output_path),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
