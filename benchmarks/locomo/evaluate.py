"""LoCoMo metric computation over runner outputs."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmarks.common.interfaces import BenchmarkEvaluator, RunnerResult
from benchmarks.common.metrics import exact_match, mean
from benchmarks.locomo.config import OUTPUT_PATH

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class LoCoMoEvaluator(BenchmarkEvaluator):
    """Compute lightweight LoCoMo metrics from runner results."""

    def evaluate(self, results: Iterable[RunnerResult]) -> dict[str, Any]:
        """Compute metrics without loading data or touching agent internals."""

        items = list(results)
        answerable = [item for item in items if item.expected_answer is not None]
        exact_matches = [
            exact_match(item.response.text, item.expected_answer or "")
            for item in answerable
        ]
        f1_scores = [f1_score(item.response.text, item.expected_answer or "") for item in answerable]
        bleu_scores = [bleu_score(item.response.text, item.expected_answer or "") for item in answerable]
        rouge_scores = [rouge_l_score(item.response.text, item.expected_answer or "") for item in answerable]
        bert_scores = [semantic_overlap_score(item.response.text, item.expected_answer or "") for item in answerable]
        diagnostics = [item.response.metadata.get("answer_diagnostics", {}) for item in items]
        latencies = [float(item.get("latency_ms", 0.0) or 0.0) for item in diagnostics]
        retrieved_counts = [len(item.get("retrieved_memory_ids", ())) for item in diagnostics]
        reflection_flags = [bool(item.get("reflection_used", False)) for item in diagnostics]
        memory_hits = [
            memory_hit(item.response.metadata.get("answer_diagnostics", {}), item.expected_answer)
            for item in answerable
        ]

        return {
            "total_results": len(items),
            "answerable_results": len(answerable),
            "exact_match": mean(exact_matches),
            "f1": mean(f1_scores),
            "bleu": mean(bleu_scores),
            "rouge_l": mean(rouge_scores),
            "bertscore": mean(bert_scores),
            "latency_ms": mean(latencies),
            "average_retrieved_memories": mean(float(count) for count in retrieved_counts),
            "reflection_rate": mean(1.0 if flag else 0.0 for flag in reflection_flags),
            "memory_hits": mean(memory_hits),
        }


def write_locomo_artifacts(
    results: Iterable[RunnerResult],
    metrics: dict[str, Any],
    output_path: Path = OUTPUT_PATH,
    provider: str = "",
    model: str = "",
    runtime_errors: int = 0,
) -> dict[str, Path]:
    """Write LoCoMo diagnostics, metrics, failures, and summary artifacts."""

    items = list(results)
    raw_path = output_path / "raw"
    metrics_path = output_path / "metrics"
    raw_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    diagnostics = [
        {
            "conversation_id": item.conversation_id,
            "question_id": item.question_id,
            **dict(item.response.metadata.get("answer_diagnostics", {})),
        }
        for item in items
    ]
    failures = [failure_record(item) for item in items if item.expected_answer and not exact_match(item.response.text, item.expected_answer)]
    summary = {
        "conversations": len({item.conversation_id for item in items}),
        "questions": len(items),
        "em": metrics.get("exact_match", 0.0),
        "f1": metrics.get("f1", 0.0),
        "latency": metrics.get("latency_ms", 0.0),
        "average_retrieval": metrics.get("average_retrieved_memories", 0.0),
        "reflection_percent": round(float(metrics.get("reflection_rate", 0.0)) * 100.0, 6),
        "memory_growth": None,
        "memory_hits": metrics.get("memory_hits", 0.0),
        "runtime_errors": runtime_errors,
        "provider": provider,
        "model": model,
        "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    paths = {
        "diagnostics": raw_path / "locomo_retrieval_diagnostics.json",
        "failures": raw_path / "locomo_failures.json",
        "single_result": raw_path / "single_conversation_benchmark_result.json",
        "metrics": metrics_path / "metrics.json",
        "summary": metrics_path / "locomo_summary.json",
        "retrieval_stage_metrics": metrics_path / "retrieval_stage_metrics.json",
        "retrieval_pipeline_trace": raw_path / "retrieval_pipeline_trace.json",
    }
    single_result = {
        "conversation_id": items[0].conversation_id if items else None,
        "result_count": len(items),
        "results": [
            {
                "conversation_id": item.conversation_id,
                "question_id": item.question_id,
                "prompt": item.prompt,
                "response_text": item.response.text,
                "expected_answer": item.expected_answer,
                "category": item.metadata.get("category"),
                "llm_used": bool(item.response.metadata.get("answer_diagnostics", {}).get("llm_used", False)),
            }
            for item in items
        ],
    }
    paths["diagnostics"].write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    paths["failures"].write_text(json.dumps(failures, indent=2), encoding="utf-8")
    paths["single_result"].write_text(json.dumps(single_result, indent=2), encoding="utf-8")
    paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["retrieval_stage_metrics"].write_text(json.dumps(retrieval_stage_metrics(items), indent=2), encoding="utf-8")
    paths["retrieval_pipeline_trace"].write_text(json.dumps(retrieval_pipeline_trace(items), indent=2), encoding="utf-8")
    return paths



def retrieval_pipeline_trace(results: Iterable[RunnerResult]) -> list[dict[str, Any]]:
    """Persist benchmark-visible retrieval traces without calling PRIMA internals."""

    traces: list[dict[str, Any]] = []
    for item in results:
        diagnostics = dict(item.response.metadata.get("answer_diagnostics", {}))
        trace = dict(diagnostics.get("retrieval_trace", {}))
        traces.append(
            {
                "conversation_id": item.conversation_id,
                "question_id": item.question_id,
                "question": item.prompt,
                "expanded_query": diagnostics.get("expanded_query") or trace.get("expanded_query"),
                "entities": diagnostics.get("entities") or trace.get("entities", []),
                "relations": diagnostics.get("relations") or trace.get("relations", []),
                "temporal_constraints": diagnostics.get("temporal_constraints") or trace.get("temporal_constraints", []),
                "dense_candidates": diagnostics.get("dense_candidates") or trace.get("dense_candidates", []),
                "sparse_candidates": diagnostics.get("sparse_candidates") or trace.get("sparse_candidates", []),
                "hybrid_candidates": trace.get("hybrid_candidates", []),
                "reranked_candidates": diagnostics.get("reranked_candidates") or trace.get("reranked_candidates", []),
                "retrieval_confidence": diagnostics.get("retrieval_confidence_components") or trace.get("retrieval_confidence", {}),
                "context_tokens": diagnostics.get("context_tokens", diagnostics.get("context_length", 0)),
                "failure_type": diagnostics.get("failure_type"),
            }
        )
    return traces


def retrieval_stage_metrics(results: Iterable[RunnerResult]) -> dict[str, Any]:
    """Compute lightweight stage metrics from public response diagnostics."""

    items = list(results)
    answerable = [item for item in items if item.expected_answer]
    return {
        "dense_recall_at_30": mean(_stage_hit(item, "dense_candidates") for item in answerable),
        "sparse_recall_at_30": mean(_stage_hit(item, "sparse_candidates") for item in answerable),
        "hybrid_recall_at_30": mean(_stage_hit(item, "hybrid_candidates") for item in answerable),
        "reranked_recall_at_5": mean(_stage_hit(item, "reranked_candidates", limit=5) for item in answerable),
        "entity_recall": mean(_entity_recall(item) for item in items),
        "temporal_recall": mean(_temporal_recall(item) for item in items),
        "relationship_recall": mean(_relationship_recall(item) for item in items),
        "average_context_tokens": mean(
            float(item.response.metadata.get("answer_diagnostics", {}).get("context_tokens", 0.0) or 0.0)
            for item in items
        ),
    }


def _stage_hit(result: RunnerResult, stage: str, limit: int = 30) -> float:
    expected_tokens = set(tokens(result.expected_answer or ""))
    if not expected_tokens:
        return 0.0
    diagnostics = dict(result.response.metadata.get("answer_diagnostics", {}))
    trace = dict(diagnostics.get("retrieval_trace", {}))
    candidates = diagnostics.get(stage) or trace.get(stage, [])
    context = " ".join(str(candidate.get("text", "")) for candidate in list(candidates)[:limit] if isinstance(candidate, dict))
    return 1.0 if expected_tokens & set(tokens(context)) else 0.0


def _entity_recall(result: RunnerResult) -> float:
    diagnostics = dict(result.response.metadata.get("answer_diagnostics", {}))
    entities = diagnostics.get("entities", [])
    if not entities:
        return 1.0
    context = " ".join(str(item.get("text", "")) for item in diagnostics.get("retrieved_memories", ()))
    lower_context = context.lower()
    return sum(1 for entity in entities if str(entity).lower() in lower_context) / max(1, len(entities))


def _temporal_recall(result: RunnerResult) -> float:
    diagnostics = dict(result.response.metadata.get("answer_diagnostics", {}))
    temporal = diagnostics.get("temporal_constraints", [])
    if not temporal:
        return 1.0
    confidence = diagnostics.get("retrieval_confidence_components", {})
    return float(confidence.get("temporal_agreement_score", 0.0) or 0.0)


def _relationship_recall(result: RunnerResult) -> float:
    diagnostics = dict(result.response.metadata.get("answer_diagnostics", {}))
    relations = diagnostics.get("relations", [])
    if not relations:
        return 1.0
    context = " ".join(str(item.get("text", "")) for item in diagnostics.get("retrieved_memories", ()))
    context_tokens = set(tokens(context))
    return sum(1 for relation in relations if str(relation).lower() in context_tokens) / max(1, len(relations))

def failure_record(result: RunnerResult) -> dict[str, Any]:
    diagnostics = dict(result.response.metadata.get("answer_diagnostics", {}))
    retrieved = diagnostics.get("retrieved_memory_ids", [])
    failure_type = diagnostics.get("failure_type")
    if not failure_type:
        if not retrieved:
            failure_type = "retrieval_miss"
        elif diagnostics.get("llm_used"):
            failure_type = "generation_error"
        else:
            failure_type = "retrieval_partial"
    return {
        "conversation_id": result.conversation_id,
        "question_id": result.question_id,
        "question": result.prompt,
        "expected": result.expected_answer,
        "prediction": result.response.text,
        "retrieved_memories": retrieved,
        "reflection_used": bool(diagnostics.get("reflection_used", False)),
        "failure_type": failure_type,
    }


def memory_hit(diagnostics: dict[str, Any], expected_answer: str | None) -> float:
    if not expected_answer:
        return 0.0
    expected_tokens = set(tokens(expected_answer))
    if not expected_tokens:
        return 0.0
    context = " ".join(str(item.get("text", "")) for item in diagnostics.get("retrieved_memories", ()))
    return 1.0 if expected_tokens & set(tokens(context)) else 0.0


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = tokens(prediction)
    ref_tokens = tokens(reference)
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu_score(prediction: str, reference: str) -> float:
    pred_tokens = tokens(prediction)
    ref_tokens = tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum(1 for token in pred_tokens if token in ref_tokens)
    precision = overlap / len(pred_tokens)
    brevity = min(1.0, math.exp(1 - len(ref_tokens) / max(1, len(pred_tokens))))
    return precision * brevity


def rouge_l_score(prediction: str, reference: str) -> float:
    pred_tokens = tokens(prediction)
    ref_tokens = tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = longest_common_subsequence(pred_tokens, ref_tokens)
    return lcs / len(ref_tokens)


def semantic_overlap_score(prediction: str, reference: str) -> float:
    return f1_score(prediction, reference)


def longest_common_subsequence(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_token == right_token else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text))]


def git_commit() -> str | None:
    git_head = Path(".git/HEAD")
    if not git_head.exists():
        return None
    head = git_head.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = Path(".git") / head.split(" ", 1)[1]
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    return head

