"""LoCoMo retrieval-only scientific validation for Retrieval V2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from benchmarks.locomo.config import DATASET_PATH
from benchmarks.locomo.loader import LoCoMoDataset
from evaluation.metrics.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank, summarize_retrieval_metrics
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType
from memory.retrieval.dense_strategy import DenseRetrievalStrategy
from memory.retrieval.reranker import Reranker
from memory.retrieval.retrieval_controller import RetrievalController
from memory.retrieval.retrieval_request import RetrievalRequest
from memory.retrieval.retrieval_result import RetrievalResult
from memory.retrieval.sparse_strategy import SparseRetrievalStrategy
from memory.retrieval.temporal_strategy import TemporalRetrievalStrategy
from runtime.context_builder import RuntimeContextBuilder, estimate_tokens

RESULTS_DIR = Path("evaluation/results")
BASELINE_PATH = RESULTS_DIR / "retrieval_optimization_results.json"


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    name: str
    strategies: tuple[Any, ...]
    use_controller: bool = True
    rerank: bool = False


def run_locomo_retrieval_validation(
    dataset_path: str | Path = DATASET_PATH,
    output_dir: str | Path = RESULTS_DIR,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run retrieval-only validation and write Phase 7.2 artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dataset_path)
    dataset_available = dataset_path.exists()
    dataset_used = dataset_path if dataset_available else Path("benchmarks/locomo/outputs/processed/locomo_smoke.json")
    conversations = list(LoCoMoDataset(dataset_used).conversations())
    records = _build_records(conversations)
    configs = _configs()
    traces_by_config = {config.name: _run_config(config, records, top_k) for config in configs}
    ablation = {name: _metrics(trace) for name, trace in traces_by_config.items()}
    full_trace = traces_by_config["full_retrieval_v2"]
    category = _category_improvements(traces_by_config)
    confidence = _confidence_validation(full_trace)
    context = _context_statistics(traces_by_config["dense"], full_trace)
    failures = _failure_breakdown(full_trace)
    baseline = _phase42_baseline()
    comparison = _comparison(baseline, ablation["full_retrieval_v2"])

    report = {
        "dataset_path": str(dataset_used),
        "dataset_available": dataset_available,
        "dataset_note": "Full LoCoMo dataset used." if dataset_available else "Full LoCoMo dataset missing; ran against local smoke fixture only.",
        "query_count": len(full_trace),
        "phase_4_2_baseline": baseline,
        "full_retrieval_v2": ablation["full_retrieval_v2"],
        "comparison": comparison,
        "ablation": ablation,
        "category_improvements": category,
        "confidence_validation": confidence,
        "context_statistics": context,
        "failure_breakdown": failures,
    }

    _write_json(output_path / "retrieval_ablation_v2.json", ablation)
    _write_json(output_path / "retrieval_category_improvements.json", category)
    _write_json(output_path / "retrieval_confidence_validation.json", confidence)
    _write_json(output_path / "retrieval_failure_breakdown.json", failures)
    _write_json(output_path / "retrieval_phase_7_2_results.json", report)
    _write_csv(output_path / "retrieval_ablation.csv", _ablation_rows(ablation, baseline))
    _write_csv(output_path / "retrieval_category.csv", _category_rows(category))
    _write_csv(output_path / "confidence_curve.csv", confidence["curve"])
    _write_csv(output_path / "context_statistics.csv", _context_rows(context))
    (output_path / "retrieval_improvement_summary.md").write_text(_summary_md(report), encoding="utf-8")
    return report


def _configs() -> tuple[ValidationConfig, ...]:
    dense = DenseRetrievalStrategy()
    sparse = SparseRetrievalStrategy()
    temporal = TemporalRetrievalStrategy()
    return (
        ValidationConfig("dense", (dense,), use_controller=False, rerank=False),
        ValidationConfig("dense_query_analysis", (dense,), use_controller=True, rerank=False),
        ValidationConfig("dense_expansion", (dense,), use_controller=True, rerank=False),
        ValidationConfig("dense_sparse", (dense, sparse), use_controller=True, rerank=False),
        ValidationConfig("dense_sparse_reranker", (dense, sparse), use_controller=True, rerank=True),
        ValidationConfig("full_retrieval_v2", (dense, sparse, temporal), use_controller=True, rerank=True),
    )


def _build_records(conversations: Iterable[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for conversation in conversations:
        repository = InMemoryMemoryRepository()
        memory_lookup: dict[str, str] = {}
        for index, turn in enumerate(conversation.turns, start=1):
            content = f"{turn.speaker}: {turn.text}"
            memory_id = f"{conversation.id}:{turn.turn_id or index}"
            note = MemoryNote.create(content=content, memory_type=MemoryType.EPISODIC, note_id=memory_id)
            repository.add(note)
            memory_lookup[memory_id] = content
        for question in conversation.questions:
            expected_ids = _expected_ids(question, memory_lookup, conversation.id)
            if not expected_ids:
                continue
            records.append(
                {
                    "conversation_id": conversation.id,
                    "question_id": question.question_id,
                    "query": question.question,
                    "answer": question.answer,
                    "category": _normalize_category(question.category, question.question),
                    "expected_memory_ids": expected_ids,
                    "repository": repository,
                    "memory_lookup": memory_lookup,
                }
            )
    return records


def _expected_ids(question: Any, memory_lookup: dict[str, str], conversation_id: str) -> list[str]:
    evidence = [str(item).strip() for item in getattr(question, "evidence", ()) if str(item).strip()]
    expected: list[str] = []
    for evidence_id in evidence:
        direct_id = f"{conversation_id}:{evidence_id}"
        if direct_id in memory_lookup:
            expected.append(direct_id)
    if expected:
        return list(dict.fromkeys(expected))
    lowered_evidence = [item.lower() for item in evidence]
    expected = [memory_id for memory_id, text in memory_lookup.items() if any(item in text.lower() for item in lowered_evidence)]
    if expected:
        return expected
    answer = str(getattr(question, "answer", "") or "").lower().strip()
    if not answer:
        return []
    return [memory_id for memory_id, text in memory_lookup.items() if answer in text.lower()]


def _run_config(config: ValidationConfig, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for record in records:
        request = RetrievalRequest(query=str(record["query"]), memory_types=(MemoryType.EPISODIC,), top_k=top_k)
        start = time.perf_counter()
        if config.use_controller:
            controller = RetrievalController(
                repository=record["repository"],
                strategies=list(config.strategies),
                reranker=Reranker(enabled=config.rerank),
            )
            response = controller.retrieve(request)
            results = list(response.results)
            confidence = response.confidence.confidence
            diagnostics = response.diagnostics
        else:
            results = list(config.strategies[0].retrieve(request, record["repository"]))[:top_k]
            confidence = _fallback_confidence(results)
            diagnostics = {}
        latency_ms = round((time.perf_counter() - start) * 1000, 6)
        retrieved_ids = [result.note.id for result in results]
        trace = {
            "conversation_id": record["conversation_id"],
            "question_id": record["question_id"],
            "query": record["query"],
            "category": record["category"],
            "expected_memory_ids": record["expected_memory_ids"],
            "retrieved_memory_ids": retrieved_ids,
            "latency_ms": latency_ms,
            "confidence": confidence,
            "rank_positions": [retrieved_ids.index(item) + 1 for item in record["expected_memory_ids"] if item in retrieved_ids],
            "context_tokens": RuntimeContextBuilder(1600).build(str(record["query"]), tuple(results)).token_count,
            "legacy_context_tokens": _legacy_context_tokens(results),
            "failure_type": _failure_type(record, retrieved_ids, diagnostics),
            "diagnostics": diagnostics,
        }
        traces.append(trace)
    return traces


def _metrics(trace: list[dict[str, Any]]) -> dict[str, float]:
    metrics = summarize_retrieval_metrics(trace)
    return {
        "recall_at_1": metrics["recall_at_1"],
        "recall_at_5": metrics["recall_at_5"],
        "mrr": metrics["mrr"],
        "ndcg_at_5": metrics["ndcg_at_5"],
        "average_retrieval_latency": metrics["average_retrieval_latency"],
        "average_retrieved_memory_count": metrics["average_retrieved_memory_count"],
    }


def _category_improvements(traces_by_config: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    categories = ("identity", "relationship", "preference", "temporal", "multi-hop")
    output: dict[str, Any] = {}
    for category in categories:
        output[category] = {}
        for name, traces in traces_by_config.items():
            subset = [trace for trace in traces if trace["category"] == category]
            output[category][name] = _metrics(subset)
        dense = output[category]["dense"]
        full = output[category]["full_retrieval_v2"]
        output[category]["improvement_vs_dense"] = {
            "recall_at_5_gain": round(full["recall_at_5"] - dense["recall_at_5"], 6),
            "mrr_gain": round(full["mrr"] - dense["mrr"], 6),
        }
    return output


def _confidence_validation(traces: list[dict[str, Any]]) -> dict[str, Any]:
    bins: list[dict[str, Any]] = []
    for lower in [index / 10 for index in range(10)]:
        upper = round(lower + 0.1, 1)
        subset = [trace for trace in traces if lower <= float(trace.get("confidence", 0.0)) < upper or (upper == 1.0 and float(trace.get("confidence", 0.0)) == 1.0)]
        metrics = _metrics(subset)
        bins.append({"bin_start": lower, "bin_end": upper, "count": len(subset), "recall_at_5": metrics["recall_at_5"], "mrr": metrics["mrr"]})
    confidences = [float(trace.get("confidence", 0.0)) for trace in traces]
    correctness = [recall_at_k(trace["expected_memory_ids"], trace["retrieved_memory_ids"], 5) for trace in traces]
    mrrs = [reciprocal_rank(trace["expected_memory_ids"], trace["retrieved_memory_ids"]) for trace in traces]
    return {
        "histogram": bins,
        "curve": bins,
        "confidence_vs_recall_at_5_correlation": _pearson(confidences, correctness),
        "confidence_vs_mrr_correlation": _pearson(confidences, mrrs),
    }


def _context_statistics(dense_trace: list[dict[str, Any]], full_trace: list[dict[str, Any]]) -> dict[str, Any]:
    before = [float(trace.get("legacy_context_tokens", 0.0)) for trace in dense_trace]
    after = [float(trace.get("context_tokens", 0.0)) for trace in full_trace]
    dense_metrics = _metrics(dense_trace)
    full_metrics = _metrics(full_trace)
    return {
        "before_retrieval_v2": _distribution(before),
        "after_retrieval_v2": _distribution(after),
        "recall_at_5_before": dense_metrics["recall_at_5"],
        "recall_at_5_after": full_metrics["recall_at_5"],
        "recall_at_5_delta": round(full_metrics["recall_at_5"] - dense_metrics["recall_at_5"], 6),
    }


def _failure_breakdown(traces: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [trace for trace in traces if recall_at_k(trace["expected_memory_ids"], trace["retrieved_memory_ids"], 5) == 0.0]
    counts: dict[str, int] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    for trace in failures:
        failure_type = str(trace.get("failure_type") or "retrieval_miss")
        counts[failure_type] = counts.get(failure_type, 0) + 1
        examples.setdefault(failure_type, [])
        if len(examples[failure_type]) < 3:
            examples[failure_type].append(
                {
                    "query": trace["query"],
                    "category": trace["category"],
                    "expected_memory_ids": trace["expected_memory_ids"],
                    "retrieved_memory_ids": trace["retrieved_memory_ids"],
                }
            )
    for name in ("retrieval_miss", "entity_resolution_failure", "temporal_failure", "relation_failure", "preference_failure", "context_truncation", "reranker_failure", "generation_failure"):
        counts.setdefault(name, 0)
        examples.setdefault(name, [])
    return {"total_failures": len(failures), "counts": counts, "representative_examples": examples}


def _failure_type(record: dict[str, Any], retrieved_ids: list[str], diagnostics: dict[str, Any]) -> str | None:
    if set(record["expected_memory_ids"]) & set(retrieved_ids[:5]):
        return None
    category = record["category"]
    if not retrieved_ids:
        return "retrieval_miss"
    if category == "relationship":
        return "relation_failure"
    if category == "preference":
        return "preference_failure"
    if category == "temporal":
        return "temporal_failure"
    if diagnostics.get("entities"):
        return "entity_resolution_failure"
    return "retrieval_miss"


def _normalize_category(category: str | None, query: str) -> str:
    category_text = str(category or "")
    text = f"{category_text} {query}".lower()
    query_text = query.lower()
    if any(term in query_text for term in ("both", "also", "same", "each other", "between", "compare")):
        return "multi-hop"
    if any(term in query_text for term in ("friend", "wife", "husband", "brother", "sister", "neighbor", "coworker", "teammate", "relationship", "partner", "family", "mother", "father", "daughter", "son")):
        return "relationship"
    if any(term in query_text for term in ("favorite", "likes", "like to", "prefers", "enjoys", "preference", "usually", "often")):
        return "preference"
    if any(term in query_text for term in ("when", "before", "after", "first", "last", "date", "year", "month", "recently", "ago", "planning on")) or category_text == "2":
        return "temporal"
    if any(term in text for term in ("multi", "hop", "graph")):
        return "multi-hop"
    return "identity"


def _phase42_baseline() -> dict[str, float]:
    if BASELINE_PATH.exists():
        payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8-sig"))
        baseline = payload.get("phase_4_1_current") or payload.get("baseline") or {}
        if baseline:
            return {key: float(baseline.get(key, 0.0)) for key in ("recall_at_1", "recall_at_5", "mrr", "ndcg_at_5")}
    return {"recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0}


def _comparison(baseline: dict[str, float], full: dict[str, float]) -> dict[str, float]:
    return {f"{key}_gain": round(float(full.get(key, 0.0)) - float(baseline.get(key, 0.0)), 6) for key in ("recall_at_1", "recall_at_5", "mrr", "ndcg_at_5")}


def _fallback_confidence(results: list[RetrievalResult]) -> float:
    if not results:
        return 0.0
    return round(sum(result.score for result in results) / len(results), 6)


def _legacy_context_tokens(results: list[RetrievalResult]) -> int:
    sections = []
    for index, result in enumerate(results, start=1):
        sections.append(
            "\n".join(
                [
                    f"[Memory {index}]",
                    f"id: {result.note.id}",
                    f"timestamp: {result.note.timestamp.isoformat()}",
                    f"importance: {result.note.salience_score}",
                    f"retrieval_score: {result.score}",
                    f"text: {result.note.content}",
                    f"metadata: {result.note.retrieval_metadata}",
                ]
            )
        )
    return estimate_tokens("\n\n".join(sections))


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"average": 0.0, "median": 0.0, "p95": 0.0}
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {"average": round(sum(values) / len(values), 6), "median": round(statistics.median(values), 6), "p95": round(ordered[index], 6)}


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    left_den = math.sqrt(sum((x - mean_left) ** 2 for x in left))
    right_den = math.sqrt(sum((y - mean_right) ** 2 for y in right))
    if left_den == 0.0 or right_den == 0.0:
        return 0.0
    return round(numerator / (left_den * right_den), 6)


def _ablation_rows(ablation: dict[str, dict[str, float]], baseline: dict[str, float]) -> list[dict[str, Any]]:
    rows = [{"configuration": "phase_4_2_baseline", **baseline}]
    rows.extend({"configuration": name, **metrics} for name, metrics in ablation.items())
    return rows


def _category_rows(category: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category_name, payload in category.items():
        for config_name, metrics in payload.items():
            if not isinstance(metrics, dict) or "recall_at_5" not in metrics:
                continue
            rows.append({"category": category_name, "configuration": config_name, **metrics})
    return rows


def _context_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"condition": "before_retrieval_v2", **context["before_retrieval_v2"], "recall_at_5": context["recall_at_5_before"]},
        {"condition": "after_retrieval_v2", **context["after_retrieval_v2"], "recall_at_5": context["recall_at_5_after"]},
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_md(report: dict[str, Any]) -> str:
    lines = [
        "# Retrieval V2 Scientific Validation",
        "",
        f"Dataset: `{report['dataset_path']}`",
        f"Dataset available: `{report['dataset_available']}`",
        f"Queries evaluated: {report['query_count']}",
        "",
        "## Phase 4.2 Baseline vs Retrieval V2",
        "",
        "| Metric | Phase 4.2 baseline | Retrieval V2 | Gain |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("recall_at_1", "recall_at_5", "mrr", "ndcg_at_5"):
        baseline = report["phase_4_2_baseline"].get(metric, 0.0)
        full = report["full_retrieval_v2"].get(metric, 0.0)
        gain = report["comparison"].get(f"{metric}_gain", 0.0)
        lines.append(f"| {metric} | {baseline:.6f} | {full:.6f} | {gain:.6f} |")
    lines.extend([
        "",
        "## Ablation Summary",
        "",
        "| Configuration | Recall@1 | Recall@5 | MRR | nDCG@5 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, metrics in report["ablation"].items():
        lines.append(f"| {name} | {metrics['recall_at_1']:.6f} | {metrics['recall_at_5']:.6f} | {metrics['mrr']:.6f} | {metrics['ndcg_at_5']:.6f} |")
    if not report["dataset_available"]:
        lines.extend(["", "Warning: full LoCoMo was not present; this report is a fixture validation, not paper-ready evidence."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoCoMo retrieval-only Phase 7.2 validation.")
    parser.add_argument("--dataset-path", default=str(DATASET_PATH))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    result = run_locomo_retrieval_validation(args.dataset_path, args.output_dir, args.top_k)
    print(json.dumps({"query_count": result["query_count"], "dataset_available": result["dataset_available"], "outputs": str(Path(args.output_dir))}, indent=2))


if __name__ == "__main__":
    main()


