import ast
import json
from pathlib import Path

import pytest

from evaluation.experiments.research_protocol import (
    action_outcomes,
    blind_order,
    evidence_checks,
    freeze_selection,
    judge_agreement,
    lifecycle_cost,
)

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_partition_keeps_related_cases_together(tmp_path):
    cases = {"a1": "a", "a2": "a", "b": "b", "c": "c"}
    partitions = {"development": ["a1", "a2"], "validation": ["b"], "final": ["c"]}
    identities = dict.fromkeys(("source", "data", "model", "config", "scorer"), "fixture")
    path = tmp_path / "selection.json"
    first = freeze_selection(path, cases, partitions, identities, {"model_calls": 0})
    assert freeze_selection(path, cases, partitions, identities, {"model_calls": 0}) == first
    with pytest.raises(ValueError, match="Frozen identity changed"):
        freeze_selection(path, cases, partitions, identities, {"model_calls": 1})
    with pytest.raises(ValueError, match="Independent units"):
        freeze_selection(path, cases, {**partitions, "development": ["a1"], "validation": ["a2", "b"]}, identities, {})


def test_evidence_membership_does_not_establish_claim_support():
    comparison = evidence_checks(["a", "b"], ["a", "b", "fake"], {"a"}, {"a", "b"}, {"a"})
    assert comparison == {"retrieval_recall": 1.0, "invalid_citations": ["fake"],
                          "unsupported_real_citations": ["b"], "supported_citations": ["a"]}
    assert blind_order("case", ["a", "b", "c"]) == blind_order("case", ["b", "c", "a"])
    recorded = lifecycle_cost(24, 6, 2, None, reuse_count=3, cache_state="warm",
                              calls=None, retries=None, input_tokens=None, output_tokens=None)
    assert recorded["amortized_construction_ms"] == 8 and recorded["generation_ms"] is None


def test_action_fixtures_report_first_final_and_ever():
    cases = json.loads((ROOT / "tests/fixtures/phase3_action_end_states.json").read_text(encoding="utf-8"))
    values = [action_outcomes(case["attempts"], case["expected"]) for case in cases]
    assert [(v["first_solved"], v["final_solved"], v["ever_solved"]) for v in values] == [
        (False, True, True), (True, False, True), (False, True, True), (False, False, False),
    ]


def test_core_does_not_import_benchmark_implementations():
    for package in ("affect", "memory", "runtime", "state", "tools", "workflow"):
        for path in (ROOT / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    assert all(alias.name.split(".")[0] != "benchmarks" for alias in node.names), path
                if isinstance(node, ast.ImportFrom):
                    assert (node.module or "").split(".")[0] != "benchmarks", path


def test_judge_calibration_discloses_disagreement_for_real_irrelevant_citations():
    adjudicated = {}
    cases = {}
    for i in range(10):
        claim = (f"person-{i}", "preferred color", "green")
        alternatives = {
            "supported": claim,
            "contradicted": (claim[0], claim[1], "blue"),
            "unrelated_real_id": (claim[0], "room color", "green"),
            "incorrect_attribution": ("neighbor", claim[1], "green"),
        }
        for variant, (label, source) in enumerate(alternatives.items()):
            case_id = f"case-{i * len(alternatives) + variant:02}"
            adjudicated[case_id] = label
            cases[case_id] = (claim, source)

    def exact_fact_judge(claim, source):
        if source[1] != claim[1]:
            return "unrelated_real_id"
        if source[0] != claim[0]:
            return "incorrect_attribution"
        return "supported" if source[2] == claim[2] else "contradicted"

    identity = {"model": "exact structured facts", "revision": "v1", "prompt": "subject/attribute/value v1"}
    order = blind_order("calibration", list(cases))
    judged = {case: exact_fact_judge(*cases[case]) for case in order}
    result = judge_agreement(adjudicated, judged, identity)
    assert result["judge"] == identity and result["cases"] == 40 and result["agreement"] == 1
    assert judged == {case: exact_fact_judge(*cases[case]) for case in reversed(order)}

    judged["case-02"] = "supported"
    result = judge_agreement(adjudicated, judged, identity)
    assert result["agreement"] == 39 / 40
    assert result["confusion"]["unrelated_real_id"] == {"supported": 1, "unrelated_real_id": 9}
