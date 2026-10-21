"""Additive report consumers for committed runtime records and legacy archives."""

from __future__ import annotations

from collections import Counter
from typing import Any

from benchmarks.goemotions.metrics import correction_metrics, same_response_eligible


def classification_report(records: list[dict[str, Any]], metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = metrics or {}
    execution = sum(bool(row.get("execution_failed")) for row in records)
    parsed = [row for row in records if not row.get("execution_failed") and row.get("parse_error") is None and row.get("baseline_parse_error") is None]
    parse_errors = len(records) - execution - len(parsed)
    scored = [row for row in parsed if isinstance(row.get("gold_labels"), (list, tuple)) and isinstance(row.get("predicted_labels"), (list, tuple))]
    wrong = sum(set(row["gold_labels"]) != set(row["predicted_labels"]) for row in scored)
    comparable = [row for row in scored if same_response_eligible(row)]
    correction = correction_metrics(
        [frozenset(row["gold_labels"]) for row in comparable],
        [frozenset(row["baseline_labels"]) for row in comparable],
        [frozenset(row["predicted_labels"]) for row in comparable],
    )
    correction["excluded_count"] = len(records) - len(comparable)
    correction["deprecated_aliases"] = {
        "improved_after_reflection": "improved_after_label_correction",
        "regressed_after_reflection": "regressed_after_label_correction",
    }
    for alias, canonical in correction["deprecated_aliases"].items():
        correction[alias] = correction[canonical]
    return {
        "primary_outcome_counts": {"execution_failure": execution, "parse_failure": parse_errors, "unscored": len(parsed) - len(scored), "label_error": wrong, "exact_match": len(scored) - wrong},
        "confidence": metrics["confidence"] if metrics.get("report_schema_version") == "2.0" else {"available": False, "mean": None, "reason": "legacy_confidence_unavailable"},
        "label_correction": correction, "runtime_reflection": {"applicable": False, "executed": 0},
        "classification_provider_calls": metrics.get("performance", {}).get("classification_provider_calls"),
        "generic_runtime_generation": "not_applicable; classification provider calls reported separately",
        "classification_component_coverage": component_coverage([row.get("runtime") for row in records]),
    }


def locomo_report(records: list[dict[str, Any]], conversations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Consume observed producer fields; last-hop legacy recall is not all-hop recall."""
    observed = [row for row in records if row.get("schema_version") == "2.0"]
    evidence_populations = {}
    for name, population in (
        ("all_observed", observed),
        ("answerable", [row for row in observed if str(row.get("category")) in {"1", "2", "3", "4"}]),
        ("adversarial", [row for row in observed if str(row.get("category")) == "5"]),
    ):
        evidence = {}
        for stage in ("candidate", "accumulated", "supplied", "cited"):
            values = [row[f"{stage}_evidence_recall"] for row in population if row.get(f"{stage}_evidence_recall") is not None]
            evidence[stage] = {"observed_count": len(values), "mean_recall": sum(values) / len(values) if values else None}
        evidence_populations[name] = evidence
    flags = Counter(key for row in observed for key, enabled in (row.get("evidence_flags") or {}).items() if enabled)
    primary = [row for row in observed if row.get("answer_correct") is True or row.get("failure_category") is not None]
    failures = Counter(str(row.get("failure_category") or "correct") for row in primary)
    return {
        "schema_version": "2.0", "qa_count": len(records), "new_schema_qa_count": len(observed),
        "evidence": evidence_populations["all_observed"], "evidence_populations": evidence_populations,
        "evidence_scope": "all observed records; answerable/adversarial populations are separate",
        "diagnostic_flags": dict(flags), "diagnostic_flags_additive": False,
        "primary_failure_counts": dict(failures), "primary_failure_observed_count": len(primary),
        "legacy_primary_failure_counts": dict(Counter(str(row.get("failure_category") or "unavailable") for row in records if row.get("schema_version") != "2.0")),
        "answer_scores": [row.get("answer_scores") for row in records],
        "conversation_replay": conversations,
        "conversation_replay_coverage": {
            "observed_conversations": len(conversations) if conversations is not None else None,
            "historical_turns": sum(row["historical_turn_count"] for row in conversations) if conversations is not None and all("historical_turn_count" in row for row in conversations) else None,
            "admitted_turns": sum(row["admitted_turn_count"] for row in conversations) if conversations is not None and all("admitted_turn_count" in row for row in conversations) else None,
            "maintenance_observed_conversations": sum("maintenance_complete" in row for row in conversations) if conversations is not None else None,
        },
        "qa_component_coverage": component_coverage([(row.get("raw_runtime_response") or {}).get("diagnostics") for row in records]),
        "replay_scope": "conversation_level_once; unavailable when producer records are absent",
        "reflection_audits": [row.get("reflection_audit") for row in records],
    }


def component_coverage(diagnostics: list[dict[str, Any] | None]) -> dict[str, Any]:
    observed = [row for row in diagnostics if isinstance(row, dict) and isinstance(row.get("executed_components"), (list, tuple))]
    return {
        "record_count": len(diagnostics), "observed_count": len(observed),
        "executed_counts": dict(Counter(name for row in observed for name in set(row["executed_components"]))) if observed else None,
    }
