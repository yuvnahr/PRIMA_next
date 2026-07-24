"""Run GoEmotions through PRIMA's existing Ollama LLM client."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmarks.goemotions.dataset import DEFAULT_DATASET_PATH, GoEmotionsExample, load_examples, load_labels
from benchmarks.goemotions.metrics import evaluate, probability_metrics, render_figures
from benchmarks.goemotions.prompts import build_prompt
from benchmarks.goemotions.systems import EncoderSystem, GoEmotionsSystem, HybridSystem, PrimaGoEmotionsSystem, PrimaQwenSystem, QwenSchemaSystem, QwenZeroShotSystem
from llm.provider import ProviderError

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
    system: str = "qwen_zero_shot",
    split: str = "test",
    sample_manifest: Path | None = None,
    write_sample_manifest: Path | None = None,
    thresholds: Path | None = None,
    calibration: Path | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run a deterministic GoEmotions test split and write standard artifacts."""

    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test.")
    dataset_path = dataset_path.with_name(f"{split}.tsv")
    labels = load_labels(dataset_path.with_name("emotions.txt"))
    examples = load_examples(dataset_path, dataset_path.with_name("emotions.txt"))
    if batch_size < 1 or parallel_workers < 1:
        raise ValueError("batch_size and parallel_workers must be positive.")
    if sample_manifest and max_samples:
        raise ValueError("Use either sample_manifest or max_samples, not both.")
    if sample_manifest:
        examples = _examples_from_manifest(examples, dataset_path, sample_manifest)
    elif max_samples > 0:
        examples = random.Random(seed).sample(examples, min(max_samples, len(examples)))  # nosec B311
    if write_sample_manifest:
        _write_sample_manifest(examples, dataset_path, split, seed, write_sample_manifest)
    _preflight_provider(provider, system, model)
    paths = _artifact_paths(output_path)
    for path in {item.parent for item in paths.values()}:
        path.mkdir(parents=True, exist_ok=True)

    active_system = _system(system, provider, model, device, batch_size, thresholds, calibration)
    previous_seed, previous_temperature = os.environ.get("PRIMA_ANSWER_SEED"), os.environ.get("PRIMA_ANSWER_TEMPERATURE")
    os.environ["PRIMA_ANSWER_SEED"] = str(seed)
    os.environ["PRIMA_ANSWER_TEMPERATURE"] = "0"
    try:
        if hasattr(active_system, "predict_many"):
            responses = active_system.predict_many([example.text for example in examples], labels)  # type: ignore[attr-defined]
            records = [_classify_response(example, labels, raw_response, details, build_prompt(example.text, labels)) for example, (raw_response, details) in zip(examples, responses, strict=True)]
            _progress_update(len(records), len(records), enabled=progress, complete=True)
        elif parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                records = list(_progress(executor.map(lambda item: _classify(item, labels, active_system), examples), len(examples), enabled=progress))
        else:
            records = [_classify(example, labels, active_system) for example in _progress(examples, len(examples), enabled=progress)]
    finally:
        _restore_environment("PRIMA_ANSWER_SEED", previous_seed)
        _restore_environment("PRIMA_ANSWER_TEMPERATURE", previous_temperature)

    gold = [frozenset(record["gold_labels"]) for record in records]
    predicted = [frozenset(record["predicted_labels"]) for record in records]
    metrics = evaluate(gold, predicted, labels)
    failures = [record for record in records if record["gold_labels"] != record["predicted_labels"]]
    metrics["parse_failure_count"] = sum(record["parse_error"] is not None for record in records)
    metrics["parse_failure_rate"] = round(metrics["parse_failure_count"] / len(records), 6) if records else 0.0
    probability_rows = [record["prediction"]["probabilities"] for record in records if isinstance(record.get("prediction"), dict)]
    metrics["probability_metrics"] = probability_metrics(gold, probability_rows, labels) if len(probability_rows) == len(records) else {"available": False, "reason": "This system emits label sets, not calibrated per-label probabilities."}
    metadata = {
        "benchmark": "goemotions",
        "dataset_path": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "dataset_samples": len(examples),
        "labels": labels,
        "runtime": {"provider": provider, "model": model, "system": active_system.name, "temperature": 0, "thinking": False, "seed": seed, "device": device, "split": split},
        "execution": {"requested_batch_size": batch_size, "actual_batch_size": batch_size if hasattr(active_system, "predict_many") else 1, "parallel_workers": parallel_workers, "note": "Ollama systems issue one request per example; encoder systems batch tensors."},
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


def _classify(example: GoEmotionsExample, labels: list[str], system: GoEmotionsSystem) -> dict[str, Any]:
    prompt = build_prompt(example.text, labels)
    raw_response, metadata = system.predict(example.text, labels)
    return _classify_response(example, labels, raw_response, metadata, prompt)


def _classify_response(example: GoEmotionsExample, labels: list[str], raw_response: str, metadata: dict[str, Any], prompt: str) -> dict[str, Any]:
    predicted, parse_error = _parse_labels(raw_response, labels)
    return {"id": example.example_id, "text": example.text, "gold_labels": sorted(example.labels), "predicted_labels": sorted(predicted), "prompt": prompt, "raw_response": raw_response, **metadata, "parse_error": parse_error}


def _preflight_provider(provider: str, system: str, model: str) -> None:
    if provider.lower() != "ollama" or system not in {"qwen_zero_shot", "qwen_schema", "prima_qwen", "goemotions_hybrid"}:
        return
    base = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as response:  # nosec B310
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise ProviderError(f"Ollama is not reachable at {base}. Start it with: ollama serve") from None
    names = {item.get("name") for item in payload.get("models", []) if isinstance(item, dict)}
    if model not in names:
        raise ProviderError(f"Ollama model '{model}' is not installed. Run: ollama pull {model}")

def _system(name: str, provider: str, model: str, device: str, batch_size: int, thresholds: Path | None = None, calibration: Path | None = None) -> GoEmotionsSystem:
    if name == "qwen_zero_shot":
        return QwenZeroShotSystem(provider, model)
    if name == "qwen_schema":
        return QwenSchemaSystem(provider, model)
    if name == "prima_qwen":
        return PrimaQwenSystem(provider, model)
    if name == "prima_goemotions":
        return PrimaGoEmotionsSystem(model if model != DEFAULT_MODEL else "SamLowe/roberta-base-go_emotions", device, batch_size, thresholds, calibration)
    if name == "goemotions_hybrid":
        return HybridSystem(model if model != DEFAULT_MODEL else "microsoft/deberta-v3-small", provider, device, batch_size, thresholds, calibration)
    if name in {"goemotions_pretrained", "goemotions_deberta"}:
        default = "SamLowe/roberta-base-go_emotions" if name == "goemotions_pretrained" else "microsoft/deberta-v3-small"
        return EncoderSystem(name, model if model != DEFAULT_MODEL else default, device, batch_size, thresholds, calibration)
    raise ValueError(f"Unsupported GoEmotions system: {name}")


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
        if "neutral" in parsed and len(parsed) > 1:
            raise ValueError("neutral cannot coexist with non-neutral labels")
        return parsed, None
    except (json.JSONDecodeError, ValueError) as exc:
        return frozenset(), str(exc)


def _examples_from_manifest(examples: list[GoEmotionsExample], dataset_path: Path, manifest_path: Path) -> list[GoEmotionsExample]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if payload.get("dataset_sha256") != expected_hash:
        raise ValueError("Sample manifest belongs to a different dataset split.")
    by_id = {example.example_id: example for example in examples}
    ids = [item["id"] for item in payload.get("samples", []) if isinstance(item, dict) and isinstance(item.get("id"), str)]
    if not ids or len(ids) != len(payload.get("samples", [])) or len(set(ids)) != len(ids):
        raise ValueError("Sample manifest must contain unique sample IDs.")
    try:
        return [by_id[example_id] for example_id in ids]
    except KeyError as exc:
        raise ValueError(f"Sample manifest ID is not in this split: {exc.args[0]}") from exc


def _write_sample_manifest(examples: list[GoEmotionsExample], dataset_path: Path, split: str, seed: int, path: Path) -> None:
    payload: dict[str, Any] = {"seed": seed, "split": split, "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(), "samples": [{"id": example.example_id, "labels": sorted(example.labels)} for example in examples]}
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)


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


def _progress(items, total: int, *, enabled: bool):
    if not enabled:
        yield from items
        return
    try:
        from tqdm import tqdm
        yield from tqdm(items, total=total, desc="GoEmotions test", unit="sample")
    except ImportError:
        import time
        started = time.monotonic()
        for index, item in enumerate(items, start=1):
            if index == total or index % max(1, total // 20) == 0:
                rate = index / max(time.monotonic() - started, 1e-6)
                print(f"GoEmotions test: {index}/{total} [{index * 100 // total}%] | {rate:.1f} samples/s", flush=True)
            yield item


def _progress_update(current: int, total: int, *, enabled: bool, complete: bool = False) -> None:
    if enabled and complete:
        print(f"GoEmotions test: {current}/{total} [100%] | batched classifier complete", flush=True)


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
    parser.add_argument("--system", choices=("qwen_zero_shot", "qwen_schema", "prima_qwen", "prima_goemotions", "goemotions_pretrained", "goemotions_deberta", "goemotions_hybrid"), default="qwen_zero_shot")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--write-sample-manifest", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    values["progress"] = not values.pop("no_progress")
    try:
        result = run_goemotions_experiment(**values)
    except ProviderError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
