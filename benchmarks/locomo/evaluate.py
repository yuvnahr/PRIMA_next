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

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class LoCoMoEvaluator(BenchmarkEvaluator):
    """Compute lightweight LoCoMo metrics from runner results."""

    def evaluate(self, results: Iterable[RunnerResult]) -> dict[str, Any]:
        """Compute metrics without loading data or touching agent internals."""

        items = list(results)
        answerable = [item for item in items if item.expected_answer is not None]
        exact_matches = [exact_match_score(item.response.text, item.expected_answer or "") for item in answerable]
        f1_scores = [f1_score(item.response.text, item.expected_answer or "") for item in answerable]
        bleu_scores = [bleu_score(item.response.text, item.expected_answer or "") for item in answerable]
        rouge_scores = [rouge_l_score(item.response.text, item.expected_answer or "") for item in answerable]
        bert_scores = bert_scores_batch(
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
    answers = [{"conversation_id": item.conversation_id, "question_id": item.question_id, "answer": item.response.text, "expected_answer": item.expected_answer, "category": item.metadata.get("category")} for item in items]
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
    from rouge import Rouge

    prediction = normalize_answer(prediction)
    reference = normalize_answer(reference)
    if not prediction or not reference:
        return 0.0
    return float(Rouge().get_scores(prediction, reference)[0]["rouge-l"]["f"])


def bert_scores_batch(predictions: list[str], references: list[str]) -> list[float]:
    if not predictions:
        return []
    from bert_score import score

    _, _, scores = score(predictions, references, lang="en", verbose=False, rescale_with_baseline=True, device="cpu")
    return [max(0.0, float(value)) for value in scores]


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

