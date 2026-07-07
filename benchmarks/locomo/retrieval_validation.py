"""LoCoMo retrieval-only scientific validation for Retrieval V2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from collections import Counter
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
    localization = _failure_localization(full_trace)
    stage_statistics = _stage_statistics(localization, len(full_trace))
    confidence_calibration = _confidence_calibration(full_trace)
    candidate_drift = _candidate_drift(full_trace)
    context_analysis = _context_analysis(full_trace)
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
        "confidence_calibration": confidence_calibration,
        "context_statistics": context,
        "context_analysis": context_analysis,
        "failure_breakdown": failures,
        "failure_localization": stage_statistics,
        "candidate_drift": candidate_drift["summary"],
    }

    _write_json(output_path / "retrieval_ablation_v2.json", ablation)
    if _env_bool("PRIMA_WRITE_FULL_RETRIEVAL_TRACE", False):
        _write_json(output_path / "retrieval_trace.json", traces_by_config)
    _write_json(output_path / "retrieval_category_improvements.json", category)
    _write_json(output_path / "retrieval_confidence_validation.json", confidence)
    _write_json(output_path / "retrieval_confidence_calibration.json", confidence_calibration)
    _write_json(output_path / "retrieval_failure_breakdown.json", failures)
    _write_json(output_path / "retrieval_failure_localization.json", localization)
    _write_json(output_path / "retrieval_stage_statistics.json", stage_statistics)
    _write_json(output_path / "candidate_drift.json", candidate_drift)
    _write_json(output_path / "context_analysis.json", context_analysis)
    _write_json(output_path / "retrieval_phase_7_2_results.json", report)
    _write_csv(output_path / "retrieval_ablation.csv", _ablation_rows(ablation, baseline))
    _write_csv(output_path / "retrieval_ablation_stage.csv", _ablation_stage_rows(traces_by_config))
    _write_csv(output_path / "retrieval_category.csv", _category_rows(category))
    _write_csv(output_path / "confidence_curve.csv", confidence["curve"])
    _write_csv(output_path / "context_statistics.csv", _context_rows(context))
    (output_path / "retrieval_improvement_summary.md").write_text(_summary_md(report), encoding="utf-8")
    (output_path / "retrieval_debug_readme.md").write_text(_debug_readme(report, output_path), encoding="utf-8")
    (output_path / "retrieval_phase7_vs_phase42.md").write_text(_phase_comparison_md(report), encoding="utf-8")
    (output_path / "retrieval_root_cause_report.md").write_text(_root_cause_report_md(report), encoding="utf-8")
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


def _failure_localization(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    localized: list[dict[str, Any]] = []
    for trace in traces:
        if recall_at_k(trace["expected_memory_ids"], trace["retrieved_memory_ids"], 5) > 0.0:
            continue
        diagnostics = dict(trace.get("diagnostics", {}))
        for expected_id in trace["expected_memory_ids"]:
            history = _candidate_rank_history(expected_id, diagnostics, trace["retrieved_memory_ids"])
            failure_stage, reason = _failure_stage(history)
            localized.append(
                {
                    "query": trace["query"],
                    "conversation_id": trace["conversation_id"],
                    "question_id": trace["question_id"],
                    "category": trace["category"],
                    "expected_memory": expected_id,
                    "expected_memory_ids": trace["expected_memory_ids"],
                    "failure_stage": failure_stage,
                    "reason": reason,
                    "candidate_rank_history": history,
                    "confidence": trace.get("confidence", 0.0),
                    "retrieved_memories": trace["retrieved_memory_ids"],
                }
            )
    return localized


def _candidate_rank_history(expected_id: str, diagnostics: dict[str, Any], retrieved_ids: list[str]) -> dict[str, Any]:
    stages = (
        ("dense", diagnostics.get("dense_top30") or diagnostics.get("dense_candidates") or []),
        ("sparse", diagnostics.get("sparse_top30") or diagnostics.get("sparse_candidates") or []),
        ("fusion", diagnostics.get("fused_top30") or diagnostics.get("hybrid_candidates") or []),
        ("reranker", diagnostics.get("reranked_top30") or diagnostics.get("reranked_candidates") or []),
        ("final", diagnostics.get("final_candidates") or [{"id": item, "score": None} for item in retrieved_ids[:5]]),
    )
    history: dict[str, Any] = {}
    previous_present = False
    previous_stage = ""
    for stage_name, candidates in stages:
        rank, score = _candidate_rank(expected_id, candidates)
        present = rank is not None
        reason_removed = ""
        if previous_present and not present:
            reason_removed = f"present_in_{previous_stage}_missing_from_{stage_name}"
        elif not present:
            reason_removed = f"not_in_{stage_name}_top{len(candidates)}"
        history[stage_name] = {
            "expected_present": present,
            "rank": rank,
            "score": score,
            "reason_removed": reason_removed,
        }
        previous_present = present
        previous_stage = stage_name
    return history


def _candidate_rank(expected_id: str, candidates: Any) -> tuple[int | None, float | None]:
    for index, candidate in enumerate(candidates or [], start=1):
        if str(candidate.get("id")) == expected_id:
            raw_score = candidate.get("score")
            score = round(float(raw_score), 6) if raw_score is not None else None
            return index, score
    return None, None


def _failure_stage(history: dict[str, Any]) -> tuple[str, str]:
    dense_present = bool(history["dense"]["expected_present"])
    sparse_present = bool(history["sparse"]["expected_present"])
    fusion_present = bool(history["fusion"]["expected_present"])
    reranker_present = bool(history["reranker"]["expected_present"])
    final_present = bool(history["final"]["expected_present"])
    if not dense_present and not sparse_present:
        return "dense_failure", "expected memory was absent from both dense_top30 and sparse_top30 candidate sources"
    if not sparse_present and dense_present:
        return "sparse_failure", "expected memory was present in dense_top30 but absent from sparse_top30"
    if (dense_present or sparse_present) and not fusion_present:
        return "fusion_failure", "expected memory entered retrieval candidates but was removed before fused_top30"
    if fusion_present and not reranker_present:
        return "reranker_failure", "expected memory was present in fused_top30 but absent after reranking"
    if reranker_present and not final_present:
        return "selection_failure", "expected memory survived reranking but was ranked below the final top5"
    return "generation_failure", "retrieval succeeded but downstream answer generation would need investigation"


def _stage_statistics(localization: list[dict[str, Any]], query_count: int) -> dict[str, Any]:
    counts = Counter(str(item["failure_stage"]) for item in localization)
    total = sum(counts.values())
    stages = ("dense_failure", "sparse_failure", "fusion_failure", "reranker_failure", "selection_failure", "context_truncation", "generation_failure")
    return {
        "query_count": query_count,
        "localized_failure_count": total,
        "stage_counts": {stage: counts.get(stage, 0) for stage in stages},
        "stage_percentages": {
            stage: round(counts.get(stage, 0) / total, 6) if total else 0.0
            for stage in stages
        },
    }


def _confidence_calibration(traces: list[dict[str, Any]]) -> dict[str, Any]:
    bins: list[dict[str, Any]] = []
    confidences = [float(trace.get("confidence", 0.0)) for trace in traces]
    successes = [1.0 if recall_at_k(trace["expected_memory_ids"], trace["retrieved_memory_ids"], 5) > 0.0 else 0.0 for trace in traces]
    mrrs = [reciprocal_rank(trace["expected_memory_ids"], trace["retrieved_memory_ids"]) for trace in traces]
    ece = 0.0
    for lower in [0.0, 0.2, 0.4, 0.6, 0.8]:
        upper = round(lower + 0.2, 1)
        subset_indexes = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < upper or (upper == 1.0 and confidence == 1.0)
        ]
        subset_traces = [traces[index] for index in subset_indexes]
        avg_confidence = _average_values(confidences[index] for index in subset_indexes)
        accuracy = _average_values(successes[index] for index in subset_indexes)
        if traces:
            ece += (len(subset_indexes) / len(traces)) * abs(avg_confidence - accuracy)
        metrics = _metrics(subset_traces)
        bins.append(
            {
                "bin_start": lower,
                "bin_end": upper,
                "count": len(subset_indexes),
                "average_confidence": avg_confidence,
                "success_rate": accuracy,
                "recall_at_5": metrics["recall_at_5"],
                "mrr": metrics["mrr"],
            }
        )
    brier = _average_values((confidence - success) ** 2 for confidence, success in zip(confidences, successes))
    return {
        "bins": bins,
        "pearson_correlation": _pearson(confidences, successes),
        "spearman_correlation": _spearman(confidences, successes),
        "expected_calibration_error": round(ece, 6),
        "brier_score": brier,
        "confidence_reliable": bool(abs(_pearson(confidences, successes)) >= 0.3 and ece <= 0.15),
    }


def _candidate_drift(traces: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [trace for trace in traces if recall_at_k(trace["expected_memory_ids"], trace["retrieved_memory_ids"], 5) == 0.0]
    examples = []
    for trace in failed[:20]:
        diagnostics = dict(trace.get("diagnostics", {}))
        examples.append(
            {
                "query": trace["query"],
                "expected_memory_ids": trace["expected_memory_ids"],
                "retrieval_v2_stage_presence": {
                    stage: any(
                        expected_id == candidate.get("id")
                        for expected_id in trace["expected_memory_ids"]
                        for candidate in diagnostics.get(stage, [])
                    )
                    for stage in ("dense_top30", "sparse_top30", "fused_top30", "reranked_top30", "final_candidates")
                },
                "phase_4_2_candidate_trace_available": False,
                "drift_assessment": "Phase 4.2 artifact contains aggregate metrics but no per-query candidate trace, so exact candidate drift cannot be proven for this query.",
            }
        )
    return {
        "summary": {
            "phase_4_2_candidate_trace_available": False,
            "reason": "Stored Phase 4.2 results in evaluation/results/retrieval_optimization_results.json contain aggregate metrics only.",
            "retrieval_v2_failed_query_count": len(failed),
            "action_required": "Regenerate Phase 4.2 with per-query stage traces before claiming exact candidate drift.",
        },
        "failed_query_examples": examples,
    }


def _context_analysis(traces: list[dict[str, Any]], token_budget: int = 1600) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    for trace in traces:
        diagnostics = dict(trace.get("diagnostics", {}))
        candidates = diagnostics.get("final_candidates", [])
        memory_rows = []
        seen_ids: set[str] = set()
        duplicate_count = 0
        for candidate in candidates:
            memory_id = str(candidate.get("id", ""))
            if memory_id in seen_ids:
                duplicate_count += 1
            seen_ids.add(memory_id)
            content = str(candidate.get("text", ""))
            metadata = {key: value for key, value in candidate.items() if key != "text"}
            memory_rows.append(
                {
                    "memory_id": memory_id,
                    "content_tokens": estimate_tokens(content),
                    "metadata_tokens": estimate_tokens(json.dumps(metadata, sort_keys=True)),
                    "total_tokens": estimate_tokens(content) + estimate_tokens(json.dumps(metadata, sort_keys=True)),
                }
            )
        context_tokens = int(trace.get("context_tokens", 0))
        per_query.append(
            {
                "query": trace["query"],
                "retrieved_count": len(candidates),
                "context_tokens": context_tokens,
                "unused_context_tokens": max(0, token_budget - context_tokens),
                "duplicate_memory_count": duplicate_count,
                "tokens_per_retrieved_memory": memory_rows,
            }
        )
    token_totals = [
        memory["total_tokens"]
        for query in per_query
        for memory in query["tokens_per_retrieved_memory"]
    ]
    return {
        "summary": {
            "query_count": len(traces),
            "average_context_tokens": _average_values(float(query["context_tokens"]) for query in per_query),
            "average_unused_context_tokens": _average_values(float(query["unused_context_tokens"]) for query in per_query),
            "average_tokens_per_retrieved_memory": _average_values(float(value) for value in token_totals),
            "duplicate_memory_count": sum(int(query["duplicate_memory_count"]) for query in per_query),
        },
        "per_query": per_query[:50],
    }


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


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    return _pearson(_ranks(left), _ranks(right))


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[index][1]:
            end += 1
        average_rank = (index + end + 2) / 2
        for ordered_index in range(index, end + 1):
            ranks[ordered[ordered_index][0]] = average_rank
        index = end + 1
    return ranks


def _average_values(values: Any) -> float:
    values_list = [float(value) for value in values]
    return round(sum(values_list) / len(values_list), 6) if values_list else 0.0


def _ablation_rows(ablation: dict[str, dict[str, float]], baseline: dict[str, float]) -> list[dict[str, Any]]:
    rows = [{"configuration": "phase_4_2_baseline", **baseline}]
    rows.extend({"configuration": name, **metrics} for name, metrics in ablation.items())
    return rows


def _ablation_stage_rows(traces_by_config: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, traces in traces_by_config.items():
        metrics = _metrics(traces)
        rows.append({"configuration": name, "stage": "final_top5", **metrics})
        if name != "full_retrieval_v2":
            continue
        for stage_name, diagnostic_key in (
            ("dense_top30", "dense_top30"),
            ("sparse_top30", "sparse_top30"),
            ("fusion_top30", "fused_top30"),
            ("reranker_top30", "reranked_top30"),
        ):
            stage_trace = []
            for trace in traces:
                retrieved_ids = [str(candidate.get("id")) for candidate in trace.get("diagnostics", {}).get(diagnostic_key, [])]
                stage_trace.append({**trace, "retrieved_memory_ids": retrieved_ids})
            rows.append({"configuration": name, "stage": stage_name, **_metrics(stage_trace)})
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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


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


def _debug_readme(report: dict[str, Any], output_path: Path) -> str:
    failures = report["failure_breakdown"]
    counts = failures.get("counts", {})
    top_failures = sorted(counts.items(), key=lambda item: int(item[1]), reverse=True)[:5]
    lines = [
        "# Retrieval V2 Debug Artifacts",
        "",
        "## Reproduce",
        "",
        "```bash",
        "git checkout retrieval-v2-debug",
        "python -m compileall memory/retrieval benchmarks/locomo runtime",
        "python -m benchmarks.locomo.retrieval_validation --dataset-path benchmarks/locomo/outputs/processed/locomo_debug_100.json --output-dir benchmarks/locomo/outputs --top-k 5",
        "python -m benchmarks.locomo.retrieval_validation --dataset-path benchmarks/locomo/external/data/locomo10.json --output-dir benchmarks/locomo/outputs/debug_full --top-k 5",
        "```",
        "",
        "## Artifacts",
        "",
        f"- Output directory: `{output_path}`",
        "- `retrieval_failure_localization.json`: failed-query stage forensics with rank history.",
        "- `retrieval_stage_statistics.json`: aggregate stage-loss counts and percentages.",
        "- `retrieval_ablation_v2.json` and `retrieval_ablation.csv`: Recall@1/5, MRR, nDCG@5, latency, and retrieved count by config.",
        "- `retrieval_ablation_stage.csv`: Recall@1/5, MRR, and nDCG@5 by ablation and stage.",
        "- `retrieval_category.csv`: per-category retrieval metrics.",
        "- `retrieval_confidence_calibration.json`: confidence bins, Pearson, Spearman, ECE, and Brier score.",
        "- `candidate_drift.json`: Retrieval V2 stage presence plus Phase 4.2 trace availability notes.",
        "- `context_analysis.json`: token and duplicate-context accounting.",
        "- `retrieval_phase7_vs_phase42.md` and `retrieval_root_cause_report.md`: scientific summaries.",
        "- `retrieval_trace.json`: optional full raw trace only when `PRIMA_WRITE_FULL_RETRIEVAL_TRACE=1`.",
        "",
        "## Current Run",
        "",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Queries evaluated: `{report['query_count']}`",
        f"- Dataset available: `{report['dataset_available']}`",
        f"- Phase 4.2 Recall@5: `{report['phase_4_2_baseline'].get('recall_at_5', 0.0):.6f}`",
        f"- Retrieval V2 Recall@5: `{report['full_retrieval_v2'].get('recall_at_5', 0.0):.6f}`",
        f"- Recall@5 gain: `{report['comparison'].get('recall_at_5_gain', 0.0):.6f}`",
        f"- Confidence vs Recall@5 correlation: `{report['confidence_validation'].get('confidence_vs_recall_at_5_correlation', 0.0):.6f}`",
        "",
        "This run is diagnostic rather than merge-ready when the baseline gains are negative. The trace should be used to drive the next repair before claiming Phase 7.2 improvement.",
        "",
        "## Failure Focus",
        "",
    ]
    if top_failures:
        lines.extend(f"- `{name}`: {count}" for name, count in top_failures)
    else:
        lines.append("- No failures recorded.")
    lines.extend(
        [
            "",
            "Use `retrieval_failure_localization.json` to inspect where each expected memory drops out. Full raw traces are opt-in because they can become too large for version control.",
        ]
    )
    return "\n".join(lines) + "\n"


def _phase_comparison_md(report: dict[str, Any]) -> str:
    lines = [
        "# Retrieval Phase 7 vs Phase 4.2",
        "",
        "| Metric | Phase 4.2 | Retrieval V2 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("recall_at_1", "recall_at_5", "mrr", "ndcg_at_5"):
        baseline = report["phase_4_2_baseline"].get(metric, 0.0)
        current = report["full_retrieval_v2"].get(metric, 0.0)
        delta = report["comparison"].get(f"{metric}_gain", 0.0)
        lines.append(f"| {metric} | {baseline:.6f} | {current:.6f} | {delta:.6f} |")
    stage_stats = report["failure_localization"]
    calibration = report["confidence_calibration"]
    lines.extend(
        [
            "",
            "## Explanation",
            "",
            f"Retrieval V2 remains below the stored Phase 4.2 aggregate baseline on this run. The largest localized failure bucket is `{_largest_stage(stage_stats)}`.",
            f"Confidence is not reliable when `confidence_reliable` is `{calibration['confidence_reliable']}` with Pearson `{calibration['pearson_correlation']:.6f}`, Spearman `{calibration['spearman_correlation']:.6f}`, ECE `{calibration['expected_calibration_error']:.6f}`, and Brier `{calibration['brier_score']:.6f}`.",
            "",
            "No additional retrieval fix is claimed here unless the metric deltas above improve. The artifact is intended to explain the regression and guide the next small repair.",
        ]
    )
    return "\n".join(lines) + "\n"


def _root_cause_report_md(report: dict[str, Any]) -> str:
    stage_stats = report["failure_localization"]
    drift = report["candidate_drift"]
    calibration = report["confidence_calibration"]
    context = report["context_analysis"]["summary"]
    largest_stage = _largest_stage(stage_stats)
    lines = [
        "# Retrieval Root Cause Report",
        "",
        "## Where Correct Memories Are First Lost",
        "",
        f"The dominant localized stage is `{largest_stage}`. Stage counts are:",
        "",
        "| Stage | Count | Share |",
        "|---|---:|---:|",
    ]
    for stage, count in stage_stats["stage_counts"].items():
        share = stage_stats["stage_percentages"].get(stage, 0.0)
        lines.append(f"| {stage} | {count} | {share:.6f} |")
    lines.extend(
        [
            "",
            "## Why They Are Lost",
            "",
            "The localization artifact records per-expected-memory rank histories across dense, sparse, fusion, reranker, and final selection. A `selection_failure` means the correct memory survives into reranker top30 but falls below final top5. A `fusion_failure` means candidate generation found it but fusion removed it from the tracked top30. A `dense_failure` here means neither initial dense nor sparse top30 contained the expected memory.",
            "",
            "## Component Most Responsible",
            "",
            f"On this run, `{largest_stage}` contributes the largest share of localized failures. See `retrieval_failure_localization.json` for the query-level evidence.",
            "",
            "## Confidence Calibration",
            "",
            f"Confidence reliability is `{calibration['confidence_reliable']}`. Pearson is `{calibration['pearson_correlation']:.6f}`, Spearman is `{calibration['spearman_correlation']:.6f}`, ECE is `{calibration['expected_calibration_error']:.6f}`, and Brier score is `{calibration['brier_score']:.6f}`. When this remains weak, confidence should not be used as a success proxy.",
            "",
            "## Candidate Drift",
            "",
            drift["reason"],
            "Exact Phase 4.2 candidate drift cannot be proven from the stored aggregate baseline alone. The current report records Retrieval V2 stage presence and explicitly marks Phase 4.2 candidate traces as unavailable.",
            "",
            "## Context Analysis",
            "",
            f"Average context tokens: `{context['average_context_tokens']:.6f}`. Average unused context tokens: `{context['average_unused_context_tokens']:.6f}`. Average tokens per retrieved memory: `{context['average_tokens_per_retrieved_memory']:.6f}`. Duplicate final memory count: `{context['duplicate_memory_count']}`.",
            "",
            "## Fixes Measurably Improved Retrieval",
            "",
            "No new algorithmic fix is introduced by this forensic pass. The instruction prohibits blind changes, so this run only localizes failure and measures calibration/context behavior.",
            "",
            "## Rejected Hypotheses",
            "",
            "Context truncation is rejected for this retrieval-only run unless `context_analysis.json` shows exhausted token budgets. Phase 4.2 candidate drift is unproven until a baseline candidate trace exists.",
            "",
            "## Future Work",
            "",
            "Regenerate Phase 4.2 with the same per-stage trace schema, then apply one small fix to the dominant failure stage and rerun the 100-query benchmark before considering full LoCoMo validation.",
        ]
    )
    return "\n".join(lines) + "\n"


def _largest_stage(stage_stats: dict[str, Any]) -> str:
    counts = stage_stats.get("stage_counts", {})
    if not counts:
        return "none"
    return max(counts.items(), key=lambda item: int(item[1]))[0]


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


