"""Official HotpotQA answer, supporting-fact, and joint metrics."""
from __future__ import annotations
import json
import re
import string
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from benchmarks.common.interfaces import BenchmarkEvaluator, BenchmarkResult

SPECIAL = {"yes", "no", "noanswer"}
METRIC_KEYS = ("em", "f1", "prec", "recall", "sp_em", "sp_f1", "sp_prec", "sp_recall", "joint_em", "joint_f1", "joint_prec", "joint_recall")

def normalize_answer(text: str) -> str:
    text = "".join(char for char in str(text).lower() if char not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())

def answer_scores(prediction: str, gold: str) -> tuple[float, float, float, float]:
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    em = float(pred == ref)
    if (pred in SPECIAL or ref in SPECIAL) and pred != ref:
        return em, 0.0, 0.0, 0.0
    pred_tokens, ref_tokens = pred.split(), ref.split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    same = sum(common.values())
    if not same:
        return em, 0.0, 0.0, 0.0
    precision, recall = same / len(pred_tokens), same / len(ref_tokens)
    return em, 2 * precision * recall / (precision + recall), precision, recall

def supporting_fact_scores(prediction: Iterable[Iterable[Any]], gold: Iterable[Iterable[Any]]) -> tuple[float, float, float, float]:
    pred, ref = {tuple(item) for item in prediction}, {tuple(item) for item in gold}
    tp, fp, fn = len(pred & ref), len(pred - ref), len(ref - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(not fp and not fn), f1, precision, recall

def project_supporting_facts(response_metadata: dict[str, Any]) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    facts, provenance = [], []
    seen = set()
    diagnostics = response_metadata.get("answer_diagnostics", {})
    selected = set(diagnostics.get("selected_memory_ids", []))
    filter_selected = diagnostics.get("structured_answer_valid") is True
    for evidence in response_metadata.get("evidence_references", []):
        if filter_selected and evidence.get("source_id") not in selected:
            continue
        source = evidence.get("provenance", {})
        title, sentence_id = source.get("source_title"), source.get("sentence_id")
        if isinstance(title, str) and isinstance(sentence_id, int) and not isinstance(sentence_id, bool) and (title, sentence_id) not in seen:
            seen.add((title, sentence_id))
            facts.append([title, sentence_id])
            provenance.append({"prediction": [title, sentence_id], "source_id": evidence.get("source_id"), "hop": evidence.get("hop"), "query": evidence.get("query")})
    return facts, provenance

def score_hotpot_record(row: BenchmarkResult | dict[str, Any]) -> dict[str, float]:
    if isinstance(row, dict):
        gold, prediction = row.get("expected_answer"), row.get("prediction", "")
        pred_sp, gold_sp = row.get("supporting_facts", []), row.get("gold_supporting_facts", [])
    else:
        gold, prediction = row.expected_answer, row.response.text
        pred_sp, _ = project_supporting_facts(row.response.metadata)
        gold_sp = row.metadata.get("question_metadata", {}).get("supporting_facts", [])
    if gold is None:
        return {}
    em, f1, prec, recall = answer_scores(prediction, gold)
    sp_em, sp_f1, sp_prec, sp_recall = supporting_fact_scores(pred_sp, gold_sp)
    joint_prec, joint_recall = prec * sp_prec, recall * sp_recall
    return {
        "em": em, "f1": f1, "prec": prec, "recall": recall,
        "sp_em": sp_em, "sp_f1": sp_f1, "sp_prec": sp_prec, "sp_recall": sp_recall,
        "joint_em": em * sp_em,
        "joint_f1": 2 * joint_prec * joint_recall / (joint_prec + joint_recall) if joint_prec + joint_recall else 0.0,
        "joint_prec": joint_prec, "joint_recall": joint_recall,
    }

def validate_predictions(predictions: dict[str, Any]) -> None:
    if set(predictions) != {"answer", "sp"} or not all(isinstance(predictions[key], dict) for key in predictions):
        raise ValueError("Hotpot predictions must contain only object fields 'answer' and 'sp'.")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in predictions["answer"].items()):
        raise ValueError("Prediction answers must map string IDs to strings.")
    for sample_id, facts in predictions["sp"].items():
        if not isinstance(sample_id, str) or not isinstance(facts, list) or any(not isinstance(fact, list) or len(fact) != 2 or not isinstance(fact[0], str) or not isinstance(fact[1], int) or isinstance(fact[1], bool) for fact in facts):
            raise ValueError("Supporting-fact predictions must be [title, integer sentence_id] lists.")

def write_predictions(predictions: dict[str, Any], path: str | Path) -> Path:
    validate_predictions(predictions)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination

class HotpotQAEvaluator(BenchmarkEvaluator):
    def evaluate(self, results: Iterable[BenchmarkResult | dict[str, Any]]) -> dict[str, float]:
        rows = list(results)
        totals = {key: 0.0 for key in METRIC_KEYS}
        scored = 0
        for row in rows:
            values = score_hotpot_record(row)
            if not values:
                continue
            scored += 1
            for key, value in values.items():
                totals[key] += value
        return {key: value / scored if scored else 0.0 for key, value in totals.items()} | {"total": len(rows), "scored": scored}