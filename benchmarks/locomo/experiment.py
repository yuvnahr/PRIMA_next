"""Configurable LoCoMo benchmark experiment runner."""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from benchmarks.common.runtime_adapter import PrimaRuntimeAdapter
from benchmarks.locomo.config import (
    LOG_PATH,
    MAX_CONVERSATIONS,
    MAX_QUESTIONS,
    MODEL,
    PARALLEL_WORKERS,
    PROVIDER,
    SEED,
    TOP_K,
)
from benchmarks.locomo.evaluate import LoCoMoEvaluator, write_locomo_artifacts
from benchmarks.locomo.loader import LoCoMoDataset
from benchmarks.locomo.runner import LoCoMoRunner


def run_locomo_experiment(
    max_conversations: int | None = None,
    max_questions: int | None = None,
    seed: int = SEED,
    top_k: int = TOP_K,
    provider: str = PROVIDER,
    model: str = MODEL,
    parallel_workers: int = PARALLEL_WORKERS,
) -> dict[str, Any]:
    """Run LoCoMo at a configurable scale and write standard artifacts."""

    conversations = list(LoCoMoDataset().conversations())
    random.Random(seed).shuffle(conversations)  # nosec B311
    max_conversations = MAX_CONVERSATIONS if max_conversations is None else max_conversations
    max_questions = MAX_QUESTIONS if max_questions is None else max_questions
    if max_conversations > 0:
        conversations = conversations[:max_conversations]
    if max_questions > 0:
        conversations = [replace(conversation, questions=conversation.questions[:max_questions]) for conversation in conversations]

    if parallel_workers > 1 and len(conversations) > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            batches = list(
                executor.map(
                    lambda conversation: run_single_conversation(conversation, top_k, provider, model),
                    conversations,
                )
            )
        results = [result for batch in batches for result in batch]
    else:
        results = []
        for conversation in conversations:
            results.extend(run_single_conversation(conversation, top_k, provider, model))

    metrics = LoCoMoEvaluator().evaluate(results)
    runtime_errors = sum(
        len(item.response.metadata.get("answer_diagnostics", {}).get("errors", ()))
        for item in results
    )
    paths = write_locomo_artifacts(
        results,
        metrics,
        provider=provider,
        model=model,
        runtime_errors=runtime_errors,
    )
    return {
        "conversations": len(conversations),
        "questions": len(results),
        "metrics": metrics,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def run_single_conversation(conversation: Any, top_k: int, provider: str, model: str) -> list[Any]:
    """Run one conversation with one isolated runtime instance."""

    agent = PrimaRuntimeAdapter(log_path=LOG_PATH / "prima_runtime_adapter.log")
    agent.answer_options = {"top_k": top_k, "provider": provider, "model": model}
    return LoCoMoRunner().run(agent, [conversation])


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
    args = parser.parse_args()

    result = run_locomo_experiment(
        max_conversations=args.max_conversations,
        max_questions=args.max_questions,
        seed=args.seed,
        top_k=args.top_k,
        provider=args.provider,
        model=args.model,
        parallel_workers=args.parallel_workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
