"""Run GoEmotions through PRIMA's existing Ollama LLM client."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmarks.goemotions.dataset import DEFAULT_DATASET_PATH, GoEmotionsExample, load_examples, load_labels
from benchmarks.goemotions.metrics import evaluate, render_figures
from benchmarks.goemotions.prompts import build_prompt
from llm.llm_client import LLMClient


OUTPUT_PATH = Path("evaluation/goemotions")
DEFAULT_MODEL = "qwen3.5:4b"


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
) -> dict[str, Any]:
    """Run a deterministic GoEmotions test split and write standard artifacts."""

    labels = load_labels(dataset_path.with_name("emotions.txt"))
    examples = load_examples(dataset_path, dataset_path.with_name("emotions.txt"))
    if batch_size < 1 or parallel_workers < 1:
        raise ValueError("batch_size and parallel_workers must be positive.")
    if max_samples > 0:
        examples = random.Random(seed).sample(examples, min(max_samples, len(examples)))  # nosec B311
    paths = _artifact_paths(output_path)
    for path in {item.parent for item in paths.values()}:
        path.mkdir(parents=True, exist_ok=True)

    previous_seed, previous_temperature = os.environ.get("PRIMA_ANSWER_SEED"), os.environ.get("PRIMA_ANSWER_TEMPERATURE")
    os.environ["PRIMA_ANSWER_SEED"] = str(seed)
    os.environ["PRIMA_ANSWER_TEMPERATURE"] = "0"
    try:
        records: list[dict[str, Any]] = []
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            if parallel_workers > 1:
                with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                    records.extend(executor.map(lambda item: _classify(item, labels, provider, model), batch))
            else:
                records.extend(_classify(example, labels, provider, model) for example in batch)
    finally:
        _restore_environment("PRIMA_ANSWER_SEED", previous_seed)
        _restore_environment("PRIMA_ANSWER_TEMPERATURE", previous_temperature)

    gold = [frozenset(record["gold_labels"]) for record in records]
    predicted = [frozenset(record["predicted_labels"]) for record in records]
    metrics = evaluate(gold, predicted, labels)
    failures = [record for record in records if record["gold_labels"] != record["predicted_labels"]]
    metadata = {
        "benchmark": "goemotions",
        "dataset_path": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "dataset_samples": len(examples),
        "labels": labels,
        "runtime": {"provider": provider, "model": model, "temperature": 0, "thinking": False, "seed": seed, "device": device},
        "execution": {"batch_size": batch_size, "parallel_workers": parallel_workers},
        "hardware": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        "git_commit": _git_commit(),
    }
    _write_json(paths["raw"], {"responses": records, "failures": failures})
    _write_json(paths["parsed"], [{key: record[key] for key in ("id", "predicted_labels", "parse_error")} for record in records])
    _write_json(paths["answers"], [{key: record[key] for key in ("id", "gold_labels", "predicted_labels")} for record in records])
    _write_json(paths["prompts"], [{key: record[key] for key in ("id", "prompt")} for record in records])
    _write_json(paths["metrics"], metrics)
    _write_json(paths["summary"], {key: metrics[key] for key in ("sample_count", "accuracy", "macro_f1", "micro_f1", "weighted_f1")})
    _write_json(paths["metadata"], metadata)
    render_figures(metrics, labels, str(paths["confusion"]), str(paths["labels"]), str(paths["predictions"]))
    paths["report"].write_text(_report(metadata, metrics, failures), encoding="utf-8")
    return {"samples": len(records), "metrics": metrics, "artifacts": {name: str(path) for name, path in paths.items()}}


def _classify(example: GoEmotionsExample, labels: list[str], provider: str, model: str) -> dict[str, Any]:
    prompt = build_prompt(example.text, labels)
    response = LLMClient(provider_name=provider).chat(prompt=prompt, model=model, temperature=0, max_tokens=128)
    predicted, parse_error = _parse_labels(response.text, labels)
    return {"id": example.example_id, "text": example.text, "gold_labels": sorted(example.labels), "predicted_labels": sorted(predicted), "prompt": prompt, "raw_response": response.text, "provider_response": response.raw, "parse_error": parse_error}


def _parse_labels(response: str, labels: list[str]) -> tuple[frozenset[str], str | None]:
    try:
        payload = json.loads(response.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        values = payload.get("labels") if isinstance(payload, dict) else None
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError("Response must contain a string labels list.")
        unknown = sorted(set(values) - set(labels))
        if unknown:
            raise ValueError(f"Unknown labels: {', '.join(unknown)}")
        parsed = frozenset(values)
        return (parsed - {"neutral"} if len(parsed) > 1 else parsed), None
    except (json.JSONDecodeError, ValueError) as exc:
        return frozenset(), str(exc)


def _artifact_paths(root: Path) -> dict[str, Path]:
    return {"raw": root / "raw" / "responses.json", "parsed": root / "parsed" / "answers.json", "answers": root / "answers" / "answers.json", "prompts": root / "prompts" / "prompts.json", "metrics": root / "metrics" / "metrics.json", "summary": root / "metrics" / "summary.json", "report": root / "reports" / "goemotions_report.md", "metadata": root / "logs" / "run_metadata.json", "confusion": root / "figures" / "confusion_matrix.png", "labels": root / "figures" / "class_distribution.png", "predictions": root / "figures" / "prediction_distribution.png"}


def _report(metadata: dict[str, Any], metrics: dict[str, Any], failures: list[dict[str, Any]]) -> str:
    rows = "\n".join(f"| {label} | {values['precision']:.6f} | {values['recall']:.6f} | {values['f1']:.6f} | {values['support']} |" for label, values in metrics["per_class"].items())
    return f"# GoEmotions Report\n\n## Configuration\n\n```json\n{json.dumps(metadata, indent=2, sort_keys=True)}\n```\n\n## Metrics\n\n- Accuracy: `{metrics['accuracy']:.6f}`\n- Macro F1: `{metrics['macro_f1']:.6f}`\n- Micro F1: `{metrics['micro_f1']:.6f}`\n- Weighted F1: `{metrics['weighted_f1']:.6f}`\n\n## Per-class metrics\n\n| Label | Precision | Recall | F1 | Support |\n| --- | ---: | ---: | ---: | ---: |\n{rows}\n\n## Failure analysis\n\n- Exact-set failures: `{len(failures)}`\n- Parse failures: `{sum(record['parse_error'] is not None for record in failures)}`\n- The confusion figure is a gold-label/predicted-label co-occurrence diagnostic for this multilabel task.\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")


def _restore_environment(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GoEmotions benchmark.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(run_goemotions_experiment(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
