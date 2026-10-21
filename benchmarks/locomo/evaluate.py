"""LoCoMo core, evidence, and optional semantic metrics."""

from __future__ import annotations

import importlib
import math
import random
import re
import string
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

from benchmarks.common.metrics import mean
from benchmarks.locomo.official_scoring import official_locomo_score, scorer_identity
from benchmarks.preflight import missing_modules

Record = dict[str, Any]


def paired_conversation_interval(values: list[tuple[str, str, float, float]], *, samples: int = 1000, seed: int = 20261020) -> Record | None:
    """Bootstrap aligned (case, conversation, left, right) observations by conversation."""
    if samples < 1 or len({row[0] for row in values}) != len(values) or any(
        not case or not conversation or not math.isfinite(a) or not math.isfinite(b) for case, conversation, a, b in values):
        raise ValueError("Paired observations must be unique with positive bootstrap samples")
    if not values:
        return None
    grouped: dict[str, list[tuple[float, float]]] = {}
    for _, conversation, a, b in values:
        grouped.setdefault(conversation, []).append((a, b))
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    ids = sorted(grouped)
    deltas = []
    for _ in range(samples):
        chosen = [pair for _ in ids for pair in grouped[rng.choice(ids)]]
        deltas.append(mean([b - a for a, b in chosen]))
    deltas.sort()
    return {"left": mean([a for _, _, a, _ in values]), "right": mean([b for _, _, _, b in values]),
        "delta": mean([b - a for _, _, a, b in values]), "paired_questions": len(values),
        "conversation_units": len(ids), "bootstrap_95_ci": [deltas[int(.025 * (samples - 1))], deltas[int(.975 * (samples - 1))]],
        "gained": sum(b > a for _, _, a, b in values), "harmed": sum(b < a for _, _, a, b in values)}


def paired_conversation_comparison(left: list[Record], right: list[Record], *, samples: int = 1000, seed: int = 20261020) -> Record:
    """Paired question scores with whole conversations as bootstrap sampling units."""
    if samples < 1 or not left or len({row["case_id"] for row in left}) != len(left) or len({row["case_id"] for row in right}) != len(right):
        raise ValueError("Paired comparison requires unique, nonempty case IDs and positive samples")
    right_by_id = {row["case_id"]: row for row in right}
    if {row["case_id"] for row in left} != set(right_by_id):
        raise ValueError("Paired case IDs differ")
    pairs = [(row, right_by_id[row["case_id"]]) for row in left]
    if any(any(a.get(key) != b.get(key) for key in ("conversation_id", "question", "expected_answer", "reference_metadata", "category")) for a, b in pairs):
        raise ValueError("Paired source/question identities differ")
    selectors: dict[str, Callable[[Record], float | bool | None]] = {
        "answerable_official_f1": lambda row: official_locomo_score(row["prediction"],
            row.get("reference_metadata", {}).get("answer", row.get("expected_answer", "")), row["category"])
            if str(row["category"]) != "5" and isinstance(row.get("prediction"), str) else None,
        "answerable_legacy_f1": lambda row: f1_score(row["prediction"], row.get("expected_answer", ""))
            if str(row["category"]) != "5" and isinstance(row.get("prediction"), str) else None,
        "adversarial_abstention_accuracy": lambda row: float(row["outcome"] == "abstained" and not row.get("execution_failed")) if str(row["category"]) == "5" and row.get("outcome") is not None else None,
        "execution_failure_rate": lambda row: float(row["execution_failed"]) if "execution_failed" in row else None,
        "invalid_citation_rate": lambda row: row.get("invalid_citations"),
        "qa_memory_write_rate": lambda row: float(row["qa_memory_write_count"] > 0) if row.get("qa_memory_write_count") is not None else None,
        "latency_ms": lambda row: row.get("latency_ms"),
    }
    metrics: Record = {}
    for name, selector in selectors.items():
        values = []
        for a, b in pairs:
            av, bv = selector(a), selector(b)
            if av is not None and bv is not None:
                values.append((a["case_id"], str(a["conversation_id"]), float(av), float(bv)))
        metrics[name] = paired_conversation_interval(values, samples=samples, seed=seed)
    required = ("answerable_official_f1", "adversarial_abstention_accuracy", "execution_failure_rate", "invalid_citation_rate", "qa_memory_write_rate")
    expected_counts = {"answerable_official_f1": sum(str(a["category"]) != "5" for a, _ in pairs),
                       "adversarial_abstention_accuracy": sum(str(a["category"]) == "5" for a, _ in pairs)}
    observed = all(metrics[name] is not None and metrics[name]["paired_questions"] == expected_counts.get(name, len(pairs)) for name in required)
    quality_pass = (observed and metrics["answerable_official_f1"]["bootstrap_95_ci"][0] > 0
        and metrics["adversarial_abstention_accuracy"]["delta"] >= 0
        and all(metrics[name]["delta"] <= 0 for name in required[2:]))
    return {"schema_version": "1.0", "sampling_unit": "conversation", "seed": seed, "samples": samples,
        "metrics": metrics, "quality_gates_pass": bool(quality_pass), "automatic_promotion": False,
        "resource_gate": "requires_fresh_integrated_measurement"}


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
        rows = [dict(row) for row in results]
        for row in rows:
            row["failure_category"] = classify_failure(row, str(row.get("runtime_profile", "prima_full")))
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
        answerable_evidence = [row for row in evidence_rows if str(row.get("category")) != "5"]
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
            "schema_version": "2.0",
            "correctness_rule": "legacy_normalized_exact_match_or_structured_category_5_abstention",
            "legacy_squad_f1": mean(f1_score(pred, ref) for pred, ref in zip(predictions, references, strict=True)),
            "smoothed_sentence_bleu": mean(bleu_score(pred, ref) for pred, ref in zip(predictions, references, strict=True)),
            "official_locomo_f1": _official_mean(scored),
            "official_locomo_scored_questions": sum(str(row.get("category")) in {"1", "2", "3", "4", "5"} for row in scored),
            "official_locomo_answerable_f1": _official_mean([row for row in scored if str(row.get("category")) != "5"]),
            "official_adversarial_raw_text_score": _official_mean([row for row in scored if str(row.get("category")) == "5"]),
            "official_scorer": scorer_identity(),
            "average_cited_evidence": _cited_count_mean(rows),
            "average_retrieved_evidence": _cited_count_mean(rows),
            "legacy_aliases": {"f1": "legacy_squad_f1", "bleu": "smoothed_sentence_bleu", "average_retrieved_evidence": "average_cited_evidence", "final_evidence_recall": "legacy citation recall"},
            "reflection_rate": mean(1.0 if row.get("reflection_interventions") else 0.0 for row in rows),
            "answer_token_coverage": _observed_mean(answerable_evidence, "answer_token_coverage"),
            "candidate_evidence_recall": _observed_mean([row for row in answerable_evidence if "candidate_source_turn_ids" in row], "candidate_evidence_recall"),
            "accumulated_evidence_recall": _observed_mean(answerable_evidence, "accumulated_evidence_recall"),
            "supplied_evidence_recall": _observed_mean(answerable_evidence, "supplied_evidence_recall"),
            "cited_evidence_recall": _observed_mean(answerable_evidence, "cited_evidence_recall"),
            "final_evidence_recall": _observed_mean(answerable_evidence, "final_evidence_recall"),
            "memory_admission_recall": _observed_mean(answerable_evidence, "memory_admission_recall"),
            "evidence_populations": {"answerable": evidence_population(answerable_evidence),
                                     "adversarial": evidence_population([row for row in evidence_rows if str(row.get("category")) == "5"]),
                                     "unannotated_questions": len(scored) - len(evidence_rows)},
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


def _answer_metrics(rows: list[Record]) -> dict[str, Any]:
    return {
        "question_count": len(rows),
        "exact_match": mean(normalized_exact_match(str(row.get("prediction", "")), str(row.get("expected_answer", ""))) for row in rows),
        "f1": mean(f1_score(str(row.get("prediction", "")), str(row.get("expected_answer", ""))) for row in rows),
        "official_locomo_f1": _official_mean(rows),
    }


def abstention_accuracy(rows: list[Record]) -> float | None:
    category_five = [row for row in rows if str(row.get("category")) == "5"]
    eligible = [row for row in category_five if row.get("outcome") is not None]
    return mean(float(row.get("outcome") == "abstained" and not row.get("execution_failed")) for row in eligible) if eligible else None


def _observed_mean(rows: list[Record], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def _cited_count_mean(rows: list[Record]) -> float | None:
    counts = [float(len(ids)) for row in rows
              if (ids := row.get("cited_source_turn_ids", row.get("final_evidence_ids"))) is not None]
    return mean(counts) if counts else None


def _official_mean(rows: list[Record]) -> float | None:
    values = [official_locomo_score(str(row.get("prediction", "")),
                                  row.get("reference_metadata", {}).get("answer", row.get("expected_answer", "")), row["category"])
              for row in rows if str(row.get("category")) in {"1", "2", "3", "4", "5"}]
    return mean(values) if values else None


def evidence_population(rows: list[Record]) -> dict[str, Any]:
    fields = ("candidate_evidence_recall", "accumulated_evidence_recall", "supplied_evidence_recall", "cited_evidence_recall")
    eligible = [row for row in rows if row.get("evidence_audit") is not None]
    return {"annotated_questions": len(rows), "diagnostics": "available" if eligible else "unavailable",
            "observed_counts": {field: sum(row.get(field) is not None for row in eligible) for field in fields},
            "recall": {field: _observed_mean(eligible, field) for field in fields},
            "flags_non_additive": {flag: sum(row.get("evidence_flags", {}).get(flag) is True for row in rows)
                                   for flag in ("admission_incomplete", "candidate_incomplete", "supplied_incomplete", "cited_incomplete")}}


def classify_failure(record: Record, profile: str) -> str | None:
    """Primary annotation-based diagnostic; correctness never depends on evidence recall."""
    if record.get("ingestion_error"):
        record["execution_failed"] = True
        return "INGESTION_FAILURE"
    if record.get("runtime_error"):
        record["execution_failed"] = True
        return "RUNTIME_FAILURE"
    if record.get("parse_error"):
        record["execution_failed"] = True
        return "ANSWER_PARSE_FAILURE"
    if record.get("execution_failed"):
        return str(record.get("failure_category") or "RUNTIME_FAILURE")
    record["correctness_rule"] = "legacy_normalized_exact_match_or_structured_category_5_abstention"
    record["answer_correct"] = False
    if str(record.get("category")) == "5" and record.get("outcome") == "abstained":
        record["answer_correct"] = True
        return None
    if not str(record.get("prediction", "")).strip():
        record["execution_failed"] = True
        return "ANSWER_PARSE_FAILURE"
    record["answer_correct"] = bool(normalized_exact_match(str(record["prediction"]), str(record.get("expected_answer", ""))))
    if record["answer_correct"]:
        return None
    if profile != "model_only" and record.get("expected_evidence"):
        if record.get("memory_admission_recall") == 0:
            return "ADMISSION_COVERAGE_GAP"
        if record.get("candidate_source_turn_ids") is not None:
            if record.get("candidate_evidence_recall") == 0:
                return "RETRIEVAL_MISS"
            if record.get("candidate_evidence_recall", 0) > 0 and record.get("supplied_evidence_recall") == 0:
                return "EVIDENCE_SELECTION_FAILURE"
    return "ANSWER_SCORING_FAILURE"


def _evidence_session_count(row: Record) -> int:
    return len({str(item).split(":", 1)[0] for item in row.get("expected_evidence", ())})


def _session_label(row: Record) -> str:
    count = _evidence_session_count(row)
    return "unannotated" if count == 0 else "multi_session" if count > 1 else "single_session"
