"""LoCoMo core, evidence, and optional semantic metrics."""

from __future__ import annotations

import importlib
import re
import string
from collections import Counter
from collections.abc import Iterable
from typing import Any

from benchmarks.common.metrics import mean
from benchmarks.preflight import missing_modules

Record = dict[str, Any]


class LoCoMoEvaluator:
    """Evaluate canonical LoCoMo question records without runtime access."""

    def __init__(
        self, *, include_rouge_l: bool = False, include_bertscore: bool = False,
        bertscore_device: str = "cpu", bertscore_batch_size: int = 16,
    ) -> None:
        if bertscore_batch_size < 1:
            raise ValueError("bertscore_batch_size must be positive")
        self.include_rouge_l = include_rouge_l
        self.include_bertscore = include_bertscore
        self.bertscore_device = bertscore_device
        self.bertscore_batch_size = bertscore_batch_size

    def evaluate(self, results: Iterable[Record]) -> dict[str, Any]:
        """Return reconciled core, category, temporal, memory, and maintenance metrics."""
        rows = list(results)
        scored = [row for row in rows if row.get("expected_answer") is not None]
        predictions = [str(row.get("prediction", "")) for row in scored]
        references = [str(row.get("expected_answer", "")) for row in scored]
        rouge = [rouge_l_score(pred, ref) for pred, ref in zip(predictions, references, strict=True)] if self.include_rouge_l else []
        bert: list[float] = []
        bert_status = "disabled"
        if self.include_bertscore:
            bert, bert_status = bert_scores_batch(
                predictions, references, device=self.bertscore_device,
                batch_size=self.bertscore_batch_size,
            )
        categories = category_metrics(scored)
        failure_counts = Counter(str(row["failure_category"]) for row in rows if row.get("failure_category"))
        conversation_growth: dict[str, int] = {}
        conversation_maintenance: dict[str, bool] = {}
        for row in rows:
            conversation_id = str(row.get("conversation_id", ""))
            conversation_growth[conversation_id] = max(
                conversation_growth.get(conversation_id, 0), int(row.get("memory_growth", 0)),
            )
            conversation_maintenance[conversation_id] = conversation_maintenance.get(conversation_id, True) and bool(
                row.get("maintenance_complete", False)
            )
        evidence_rows = [row for row in scored if row.get("expected_evidence")]
        category_total = sum(int(item["question_count"]) for item in categories.values())
        if category_total != len(scored):
            raise ValueError("LoCoMo category totals do not reconcile")
        failed_questions = sum(bool(row.get("execution_failed")) for row in rows)
        evaluation_failures = sum(
            bool(row.get("failure_category")) and not bool(row.get("execution_failed")) for row in rows
        )
        if sum(failure_counts.values()) != failed_questions + evaluation_failures:
            raise ValueError("LoCoMo failure totals do not reconcile")
        return {
            "total_questions": len(rows),
            "scored_questions": len(scored),
            "failed_questions": failed_questions,
            "evaluation_failure_questions": evaluation_failures,
            "exact_match": mean(normalized_exact_match(pred, ref) for pred, ref in zip(predictions, references, strict=True)),
            "f1": mean(f1_score(pred, ref) for pred, ref in zip(predictions, references, strict=True)),
            "bleu": mean(bleu_score(pred, ref) for pred, ref in zip(predictions, references, strict=True)),
            "rouge_l": mean(rouge) if rouge else None,
            "rouge_l_status": "available" if self.include_rouge_l else "disabled",
            "bertscore": mean(bert) if bert else None,
            "bertscore_status": bert_status,
            "average_latency_ms": mean(float(row.get("latency_ms", 0.0)) for row in rows),
            "average_retrieved_evidence": mean(float(len(row.get("final_evidence_ids", ()))) for row in rows),
            "reflection_rate": mean(1.0 if row.get("reflection_interventions") else 0.0 for row in rows),
            "answer_token_coverage": mean(float(row.get("answer_token_coverage", 0.0)) for row in evidence_rows),
            "candidate_evidence_recall": mean(float(row.get("candidate_evidence_recall", 0.0)) for row in evidence_rows),
            "final_evidence_recall": mean(float(row.get("final_evidence_recall", 0.0)) for row in evidence_rows),
            "memory_admission_recall": mean(float(row.get("memory_admission_recall", 0.0)) for row in evidence_rows),
            "memory_growth": sum(conversation_growth.values()),
            "maintenance_completion_rate": mean(1.0 if value else 0.0 for value in conversation_maintenance.values()),
            "abstention_category_5_accuracy": abstention_accuracy(scored),
            "category_metrics": categories,
            "temporal_breakdown": breakdown(scored, lambda row: "temporal" if str(row.get("category")) == "2" else "non_temporal"),
            "multi_session_breakdown": breakdown(scored, _session_label),
            "failure_taxonomy": {
                "total_failures": sum(failure_counts.values()),
                "execution_failures": failed_questions,
                "evaluation_failures": evaluation_failures,
                "counts": dict(sorted(failure_counts.items())),
            },
        }


def normalize_answer(text: str) -> str:
    """Apply SQuAD normalization: lowercase, punctuation/articles removal, whitespace collapse."""
    lowered = str(text).lower()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def normalized_exact_match(prediction: str, reference: str) -> float:
    """Compare complete normalized strings without discarding order or multiplicity."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def exact_match_score(prediction: str, reference: str) -> float:
    """Compatibility name for normalized exact-string comparison."""
    return normalized_exact_match(prediction, reference)


def f1_score(prediction: str, reference: str) -> float:
    """Compute normalized token F1 while preserving duplicate-token multiplicity."""
    predicted, expected = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def bleu_score(prediction: str, reference: str) -> float:
    """Compute the existing sentence BLEU diagnostic from normalized tokens."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    predicted, expected = normalize_answer(prediction).split(), normalize_answer(reference).split()
    return float(sentence_bleu([expected], predicted, smoothing_function=SmoothingFunction().method1)) if predicted and expected else 0.0


def rouge_l_score(prediction: str, reference: str) -> float:
    """Compute deterministic token ROUGE-L F1 without an optional package."""
    predicted, expected = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    previous = [0] * (len(expected) + 1)
    for token in predicted:
        current = [0]
        for index, expected_token in enumerate(expected, 1):
            current.append(previous[index - 1] + 1 if token == expected_token else max(previous[index], current[-1]))
        previous = current
    common = previous[-1]
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall) if common else 0.0


def bert_scores_batch(
    predictions: list[str], references: list[str], *, device: str, batch_size: int,
) -> tuple[list[float], str]:
    """Run opt-in BERTScore with explicit device/batch configuration."""
    missing = missing_modules(("bert_score",))
    if missing:
        return [], f"unavailable: {', '.join(missing)}"
    if not predictions:
        return [], "available"
    score = importlib.import_module("bert_score").score

    _, _, scores = score(
        predictions, references, lang="en", verbose=False, rescale_with_baseline=True,
        device=device, batch_size=batch_size,
    )
    return [float(value) for value in scores], "available"


def evidence_recall(expected: Iterable[str], found: Iterable[str]) -> float:
    """Return exact evidence-ID recall, or zero when no annotation exists."""
    gold, actual = set(expected), set(found)
    return len(gold & actual) / len(gold) if gold else 0.0


def answer_token_coverage(answer: str, evidence_texts: Iterable[str]) -> float:
    """Report answer-token coverage separately from evidence-ID recall."""
    expected = set(normalize_answer(answer).split())
    observed = set(normalize_answer(" ".join(evidence_texts)).split())
    return len(expected & observed) / len(expected) if expected else 0.0


def category_metrics(rows: list[Record]) -> dict[str, dict[str, float | int]]:
    output = {}
    for category in sorted({str(row.get("category", "")) for row in rows}):
        subset = [row for row in rows if str(row.get("category", "")) == category]
        output[category] = _answer_metrics(subset)
    return output


def breakdown(rows: list[Record], key: Any) -> dict[str, dict[str, float | int]]:
    labels = [str(key(row)) for row in rows]
    return {label: _answer_metrics([row for row, row_label in zip(rows, labels, strict=True) if row_label == label]) for label in sorted(set(labels))}


def _answer_metrics(rows: list[Record]) -> dict[str, float | int]:
    return {
        "question_count": len(rows),
        "exact_match": mean(normalized_exact_match(str(row.get("prediction", "")), str(row.get("expected_answer", ""))) for row in rows),
        "f1": mean(f1_score(str(row.get("prediction", "")), str(row.get("expected_answer", ""))) for row in rows),
    }


def abstention_accuracy(rows: list[Record]) -> float:
    category_five = [row for row in rows if str(row.get("category")) == "5"]
    return mean(
        1.0 if row.get("outcome") == "abstained" or normalized_exact_match(
            str(row.get("prediction", "")), str(row.get("expected_answer", "")),
        ) else 0.0
        for row in category_five
    )


def _evidence_session_count(row: Record) -> int:
    return len({str(item).split(":", 1)[0] for item in row.get("expected_evidence", ())})


def _session_label(row: Record) -> str:
    count = _evidence_session_count(row)
    return "unannotated" if count == 0 else "multi_session" if count > 1 else "single_session"
