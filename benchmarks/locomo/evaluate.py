"""LoCoMo metric computation over runner outputs."""

from __future__ import annotations

import json
import platform
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmarks.common.interfaces import BenchmarkEvaluator, RunnerResult
from benchmarks.common.metrics import mean
from benchmarks.locomo.config import OUTPUT_PATH
from benchmarks.preflight import missing_modules

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class LoCoMoEvaluator(BenchmarkEvaluator):
    """Compute lightweight LoCoMo metrics from runner results."""

    def __init__(self, *, include_bertscore: bool = False) -> None:
        self.include_bertscore = include_bertscore

    def evaluate(self, results: Iterable[RunnerResult]) -> dict[str, Any]:
        """Compute metrics without loading data or touching agent internals."""

        items = list(results)
        answerable = [item for item in items if item.expected_answer is not None]
        exact_matches = [exact_match_score(item.response.text, item.expected_answer or "") for item in answerable]
        f1_scores = [f1_score(item.response.text, item.expected_answer or "") for item in answerable]
        bleu_scores = [bleu_score(item.response.text, item.expected_answer or "") for item in answerable]
        rouge_scores = [rouge_l_score(item.response.text, item.expected_answer or "") for item in answerable]
        bert_scores: list[float] = []
        bertscore_status = "disabled"
        if self.include_bertscore:
            bert_scores, bertscore_status = bert_scores_batch(
                [item.response.text for item in answerable],
                [item.expected_answer or "" for item in answerable],
            )
        diagnostics = [item.response.metadata.get("answer_diagnostics", {}) for item in items]
        latencies = [float(item.get("latency_ms", 0.0) or 0.0) for item in diagnostics]
        retrieved_counts = [len(item.get("retrieved_memory_ids", ())) for item in diagnostics]
        reflection_flags = [bool(item.get("reflection_used", False)) for item in diagnostics]
        memory_hits = [
            memory_hit(item.response.metadata.get("answer_diagnostics", {}), item.expected_answer)
            for item in answerable
        ]
        evidence = [summary for item in answerable if (summary := evidence_summary(item)) is not None]

        return {
            "total_results": len(items),
            "answerable_results": len(answerable),
            "exact_match": mean(exact_matches),
            "f1": mean(f1_scores),
            "bleu": mean(bleu_scores),
            "rouge_l": mean(rouge_scores),
            "bertscore": mean(bert_scores) if bert_scores else None,
            "bertscore_status": bertscore_status,
            "latency_ms": mean(latencies),
            "average_retrieved_memories": mean(float(count) for count in retrieved_counts),
            "reflection_rate": mean(1.0 if flag else 0.0 for flag in reflection_flags),
            "memory_hits": mean(memory_hits),
            **evidence_metrics(evidence),
            "category_metrics": category_metrics(answerable),
            "failure_taxonomy": failure_taxonomy(answerable),
        }


def write_locomo_artifacts(
    results: Iterable[RunnerResult],
    metrics: dict[str, Any],
    output_path: Path = OUTPUT_PATH,
    provider: str = "",
    model: str = "",
    runtime_errors: int = 0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the production LoCoMo QA artifact set."""

    items = list(results)
    paths_by_kind = {name: output_path / name for name in ("raw", "parsed", "metrics", "reports", "logs", "answers", "prompts")}
    for path in paths_by_kind.values():
        path.mkdir(parents=True, exist_ok=True)

    diagnostics = [
        {
            "conversation_id": item.conversation_id,
            "question_id": item.question_id,
            "category": item.metadata.get("category"),
            "expected_evidence": list(item.metadata.get("evidence", ())),
            **dict(item.response.metadata.get("answer_diagnostics", {})),
        }
        for item in items
    ]
    failures = [failure_record(item) for item in items if item.expected_answer and not exact_match_score(item.response.text, item.expected_answer)]
    provenance = {
        "backend_fingerprint": metadata.get("backend_fingerprint") if metadata else None,
        "benchmark": metadata or {},
        "hardware": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        "model": {"provider": provider, "model": model, "thinking": False},
        "runtime": {"python": sys.version, "implementation": platform.python_implementation()},
        "seed": (metadata or {}).get("seed"),
        "benchmark_version": (metadata or {}).get("benchmark_version"),
    }
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
        "candidate_evidence_recall": metrics.get("candidate_evidence_recall", 0.0),
        "final_context_evidence_recall": metrics.get("final_context_evidence_recall", 0.0),
        "final_all_evidence_rate": metrics.get("final_all_evidence_rate", 0.0),
        "runtime_errors": runtime_errors,
        "provider": provider,
        "model": model,
        "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance,
    }

    paths = {
        "raw": paths_by_kind["raw"] / "responses.json",
        "parsed": paths_by_kind["parsed"] / "answers.json",
        "answers": paths_by_kind["answers"] / "answers.json",
        "prompts": paths_by_kind["prompts"] / "prompts.json",
        "metrics": paths_by_kind["metrics"] / "metrics.json",
        "summary": paths_by_kind["metrics"] / "locomo_summary.json",
        "report": paths_by_kind["reports"] / "locomo_qa_report.md",
        "log": paths_by_kind["logs"] / "run_metadata.json",
    }
    answers = [{"conversation_id": item.conversation_id, "question_id": item.question_id, "answer": item.response.text, "expected_answer": item.expected_answer, "category": item.metadata.get("category"), "evidence": list(item.metadata.get("evidence", ()))} for item in items]
    parsed = [{"conversation_id": item.conversation_id, "question_id": item.question_id, "raw_response": item.response.metadata.get("answer_diagnostics", {}).get("raw_response"), "parsed_answer": item.response.text, "errors": item.response.metadata.get("answer_diagnostics", {}).get("errors", [])} for item in items]
    prompts = [{"conversation_id": item.conversation_id, "question_id": item.question_id, "prompt": item.response.metadata.get("answer_diagnostics", {}).get("prompt")} for item in items]
    paths["raw"].write_text(json.dumps({"responses": diagnostics, "failures": failures}, indent=2), encoding="utf-8")
    paths["parsed"].write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    paths["answers"].write_text(json.dumps(answers, indent=2), encoding="utf-8")
    paths["prompts"].write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["log"].write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    paths["report"].write_text("# LoCoMo QA Report\n\n" + "\n".join(f"- {key}: {value}" for key, value in metrics.items()) + f"\n- failures: {len(failures)}\n", encoding="utf-8")
    return paths



def failure_record(result: RunnerResult) -> dict[str, Any]:
    diagnostics = dict(result.response.metadata.get("answer_diagnostics", {}))
    retrieved = diagnostics.get("retrieved_memory_ids", [])
    summary = evidence_summary(result)
    return {
        "conversation_id": result.conversation_id,
        "question_id": result.question_id,
        "question": result.prompt,
        "expected": result.expected_answer,
        "prediction": result.response.text,
        "retrieved_memories": retrieved,
        "reflection_used": bool(diagnostics.get("reflection_used", False)),
        "failure_type": localized_failure_type(result, summary),
        "evidence_localization": summary,
    }


FAILURE_TYPES = (
    "A_gold_absent_candidate_pool",
    "B_gold_lost_ranking_or_selection",
    "C_evidence_present_llm_error",
    "D_evaluation_penalty",
    "E_temporal_reasoning",
    "F_multi_memory_aggregation",
    "G_query_understanding_or_expansion",
    "H_other_ambiguous",
)


def evidence_summary(result: RunnerResult) -> dict[str, Any] | None:
    evidence = {str(item) for item in result.metadata.get("evidence", ()) if str(item)}
    category = str(result.metadata.get("category", ""))
    if not evidence or category == "5":
        return None
    diagnostics = dict(result.response.metadata.get("answer_diagnostics", {}))
    stages = dict(diagnostics.get("retrieval_stages", {}))
    stored = {str(item) for item in diagnostics.get("stored_source_turn_ids", ())}
    dense = _source_ids(stages.get("dense_top30", ()))
    sparse = _source_ids(stages.get("sparse_top30", ()))
    fused = _source_ids(stages.get("fused_top30", ()))
    reranked = _source_ids(stages.get("reranked_top30", ()))
    final_items = list(stages.get("final_candidates", ()))
    final = _source_ids(final_items)
    ranks = [
        index
        for index, item in enumerate(final_items, start=1)
        if str(item.get("source_turn_id", "")) in evidence
    ]

    def recall(found: set[str]) -> float:
        return len(evidence & found) / len(evidence)

    candidate = dense | sparse
    return {
        "gold_evidence_count": len(evidence),
        "stored_evidence_recall": recall(stored),
        "dense_evidence_recall": recall(dense),
        "sparse_evidence_recall": recall(sparse),
        "candidate_evidence_recall": recall(candidate),
        "fusion_evidence_recall": recall(fused),
        "reranked_evidence_recall": recall(reranked),
        "final_context_evidence_recall": recall(final),
        "final_all_evidence": evidence <= final,
        "final_relevant_ranks": ranks,
    }


def evidence_metrics(summaries: list[dict[str, Any]]) -> dict[str, float]:
    ranks = [float(rank) for summary in summaries for rank in summary["final_relevant_ranks"]]
    keys = (
        "stored_evidence_recall",
        "dense_evidence_recall",
        "sparse_evidence_recall",
        "candidate_evidence_recall",
        "fusion_evidence_recall",
        "reranked_evidence_recall",
        "final_context_evidence_recall",
    )
    return {
        **{key: mean(float(summary[key]) for summary in summaries) for key in keys},
        "final_all_evidence_rate": mean(1.0 if summary["final_all_evidence"] else 0.0 for summary in summaries),
        "average_relevant_final_rank": mean(ranks),
        "evidence_scored_questions": float(len(summaries)),
    }


def category_metrics(results: list[RunnerResult]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for category in sorted({str(item.metadata.get("category", "")) for item in results}):
        rows = [item for item in results if str(item.metadata.get("category", "")) == category]
        evidence = [summary for item in rows if (summary := evidence_summary(item)) is not None]
        output[category] = {
            "question_count": len(rows),
            "exact_match": mean(exact_match_score(item.response.text, item.expected_answer or "") for item in rows),
            "f1": mean(f1_score(item.response.text, item.expected_answer or "") for item in rows),
            "memory_hits": mean(memory_hit(item.response.metadata.get("answer_diagnostics", {}), item.expected_answer) for item in rows),
            **evidence_metrics(evidence),
        }
    return output


def failure_taxonomy(results: list[RunnerResult]) -> dict[str, Any]:
    failures = [
        localized_failure_type(item)
        for item in results
        if not exact_match_score(item.response.text, item.expected_answer or "")
    ]
    counts = Counter(failures)
    total = len(failures)
    return {
        "total_failures": total,
        "counts": {name: counts.get(name, 0) for name in FAILURE_TYPES},
        "percentages": {name: (counts.get(name, 0) / total if total else 0.0) for name in FAILURE_TYPES},
    }


def localized_failure_type(result: RunnerResult, summary: dict[str, Any] | None = None) -> str:
    diagnostics = dict(result.response.metadata.get("answer_diagnostics", {}))
    if diagnostics.get("errors"):
        return "H_other_ambiguous"
    if f1_score(result.response.text, result.expected_answer or "") >= 0.8:
        return "D_evaluation_penalty"
    summary = summary if summary is not None else evidence_summary(result)
    if summary is None:
        return "H_other_ambiguous"
    category = str(result.metadata.get("category", ""))
    if summary["candidate_evidence_recall"] < 1.0:
        return "A_gold_absent_candidate_pool"
    if summary["final_context_evidence_recall"] < 1.0:
        return "F_multi_memory_aggregation" if category == "1" else "B_gold_lost_ranking_or_selection"
    if category == "2":
        return "E_temporal_reasoning"
    if category == "1":
        return "F_multi_memory_aggregation"
    return "C_evidence_present_llm_error"


def _source_ids(candidates: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(item["source_turn_id"])
        for item in candidates
        if item.get("source_turn_id")
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
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()
    pred_tokens = [stemmer.stem(token) for token in normalize_answer(prediction).split()]
    ref_tokens = [stemmer.stem(token) for token in normalize_answer(reference).split()]
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu_score(prediction: str, reference: str) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    return float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=SmoothingFunction().method1)) if pred_tokens and ref_tokens else 0.0


def rouge_l_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    previous = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for index, ref_token in enumerate(ref_tokens, start=1):
            current.append(previous[index - 1] + 1 if pred_token == ref_token else max(previous[index], current[-1]))
        previous = current
    common = previous[-1]
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def bert_scores_batch(predictions: list[str], references: list[str]) -> tuple[list[float], str]:
    missing = missing_modules(("bert_score",))
    if missing:
        return [], f"unavailable: {', '.join(missing)}"
    if not predictions:
        return [], "available"
    from bert_score import score

    _, _, scores = score(predictions, references, lang="en", verbose=False, rescale_with_baseline=True, device="cpu")
    return [max(0.0, float(value)) for value in scores], "available"


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text))]


def normalize_answer(text: str) -> str:
    words = re.sub(r"[^\w\s]", "", str(text).lower().replace(",", "")).split()
    return " ".join(word for word in words if word not in {"a", "an", "the", "and"})


def exact_match_score(prediction: str, reference: str) -> float:
    return float(set(normalize_answer(prediction).split()) == set(normalize_answer(reference).split()))


def git_commit() -> str | None:
    git_head = Path(".git/HEAD")
    if not git_head.exists():
        return None
    head = git_head.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = Path(".git") / head.split(" ", 1)[1]
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    return head

