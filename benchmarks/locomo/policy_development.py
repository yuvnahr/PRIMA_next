"""Frozen one-factor CPU policy analyses; no answer model or automatic promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.common import atomic_write_json
from benchmarks.locomo.evaluate import paired_conversation_interval
from benchmarks.locomo.experiment import IngestionPolicy, fingerprint
from benchmarks.locomo.retrieval_development import run_development


def split_identity(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_policy_development(dataset: Path, split: Path, controls: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads(split.read_text(encoding="utf-8"))
    frozen = json.loads(controls.read_text(encoding="utf-8"))
    if frozen["dataset_sha256"] != fingerprint(dataset) or frozen["split_sha256"] != split_identity(manifest):
        raise ValueError("Frozen policy dataset/split identity differs")
    if (output / "policy_report.json").exists():
        raise ValueError("Use a fresh policy output namespace")
    cache: dict[tuple[Any, ...], Any] = {}
    report: dict[str, Any] = {"controls_sha256": fingerprint(controls), "stages": {}, "production_promotion": False}
    traces: dict[str, list[dict[str, Any]]] = {}
    for name, stage in frozen["cpu_stages"].items():
        destination = output / name
        measured = run_development(dataset, manifest, {name: stage["runtime_config"]}, destination,
            ingestion_policy=IngestionPolicy(stage["ingestion_policy"]), include_captions=stage["include_captions"],
            gate_diagnostic=stage.get("gate_diagnostic", False), _replay_cache=cache)
        ledger = measured["ledger"][name]
        if ledger["status"] == "externally_pending":
            report["stages"][name] = ledger
            continue
        rows = [json.loads(line) for line in (destination / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
        traces[name] = rows
        ledger["admission_recall"] = (sum(row["admission_recall"] for row in rows if row["category"] != "5" and row["admission_recall"] is not None)
                                      / ledger["annotated_answerable_count"] if ledger["annotated_answerable_count"] else None)
        if stage.get("gate_diagnostic"):
            audits = [row["reflection_audit"] for row in rows]
            ledger["gate_diagnostics"] = {key: sum(row[key] for row in audits)
                for key in ("heuristic_recommended", "gate_authorized", "gate_suppressed", "engine_executed")}
        report["stages"][name] = {"status": "measured_cpu_only_unpromoted", "summary": ledger,
            "report_sha256": fingerprint(destination / "development_report.json"), "trace_sha256": fingerprint(destination / "traces.jsonl")}
        atomic_write_json(output / "policy_report.json", report)
    # These comparisons quantify retrieval/admission proxies, never answer-model quality.
    report["paired_retrieval_proxies"] = {}
    baseline = traces.get("normal", [])
    for name in ("controlled", "factual_candidate", "captions"):
        if not baseline or name not in traces:
            report["paired_retrieval_proxies"][name] = {"status": "externally_pending"}
            continue
        right = {row["case_id"]: row for row in traces[name]}
        recalls, latencies = [], []
        for row in baseline:
            other = right[row["case_id"]]
            identity = (row["case_id"], row["case_id"].split(":", 1)[0])
            if row["category"] != "5" and row["stage_recall"]["returned"] is not None:
                recalls.append((*identity, row["stage_recall"]["returned"], other["stage_recall"]["returned"]))
            latencies.append((*identity, row["audit"]["elapsed_ms"], other["audit"]["elapsed_ms"]))
        report["paired_retrieval_proxies"][name] = {
            "returned_recall": paired_conversation_interval(recalls, samples=frozen["bootstrap_samples"], seed=frozen["seed"]),
            "latency_ms": paired_conversation_interval(latencies, samples=frozen["bootstrap_samples"], seed=frozen["seed"]),
            "sampling_unit": "conversation", "answer_quality_measured": False}
    report["experiment_disposition"] = frozen["experiment_disposition"]
    atomic_write_json(output / "policy_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_policy_development(args.dataset, args.split, args.controls, args.output)
    print(json.dumps({name: {"status": stage["status"], "questions": stage["summary"]["question_count"]}
                      for name, stage in report["stages"].items()}, indent=2))


if __name__ == "__main__":
    main()
