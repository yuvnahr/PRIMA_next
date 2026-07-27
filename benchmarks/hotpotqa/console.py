"""Plain-text terminal reporting for HotpotQA runs."""
from __future__ import annotations
import shutil
import textwrap
from collections import Counter
from typing import Any
from benchmarks.hotpotqa.evaluate import score_hotpot_record

class HotpotQATerminalReporter:
    def __init__(self, *, quiet: bool = False, progress: bool = False) -> None:
        self.enabled = not quiet
        self.show_questions = self.enabled and progress
        self.width = max(40, min(80, shutil.get_terminal_size((60, 20)).columns))

    def _rule(self) -> None:
        print("=" * self.width)

    def _field(self, label: str, value: Any) -> None:
        print(f"{label:<16}: {value if value is not None else '-'}")

    def _wrapped(self, heading: str, value: Any) -> None:
        print(f"\n{heading}\n{textwrap.fill(str(value or '-'), self.width)}")

    def header(self, config: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._rule(); print("HotpotQA Run")
        for label, key in (("Dataset set", "dataset_set"), ("Benchmark mode", "mode"), ("Dataset path", "dataset_path"), ("Selected samples", "sample_count"), ("Sampling", "sampling_strategy"), ("Resolved seed", "resolved_seed"), ("Provider", "provider"), ("Model", "model"), ("Reasoning mode", "reasoning_mode"), ("Top-k", "top_k"), ("Max hops", "max_hops"), ("Output directory", "output_dir"), ("Resume", "resume")):
            self._field(label, config.get(key, "-"))
        self._rule()

    def result(self, index: int, total: int, record: dict[str, Any]) -> None:
        if not self.show_questions:
            return
        self._rule(); print(f"HotpotQA Result {index}/{total}"); self._rule()
        self._field("Sample ID", record.get("sample_id")); self._field("Type", record.get("type") or "-"); self._field("Level", record.get("level") or "-")
        self._wrapped("Question", record.get("question"))
        answer_em = 0.0
        if record.get("runtime_error"):
            self._field("Runtime status", "failed"); self._field("Failure stage", record.get("failure_stage") or "unknown"); self._field("Failure category", record.get("failure_category") or "UNKNOWN")
            self._wrapped("Error", record.get("runtime_error")); self._field("Latency", f"{record.get('total_latency', 0.0):.2f} seconds")
        else:
            self._wrapped("Prediction", record.get("prediction")); self._wrapped("Reference", record.get("expected_answer"))
            scores = score_hotpot_record(record)
            answer_em = scores.get("em", 0.0)
            print("\nScores")
            for label, key in (("Answer EM", "em"), ("Answer F1", "f1"), ("Supporting EM", "sp_em"), ("Supporting F1", "sp_f1"), ("Joint EM", "joint_em"), ("Joint F1", "joint_f1")):
                self._field(label, f"{scores.get(key, 0.0):.2%}")
            print("\nExecution")
            for label, key in (("Runtime status", "runtime_status"), ("Stop reason", "final_stop_reason"), ("Reasoning hops", "reasoning_hops"), ("Retrieval calls", "retrieval_calls"), ("Reflections", "reflection_interventions"), ("Evidence items", "evidence_count")):
                self._field(label, record.get(key, "-") if key != "runtime_status" else "completed")
            self._field("Failure category", record.get("failure_category") or "none")
            self._field("Latency", f"{record.get('total_latency', 0.0):.2f} seconds")
        print("\n✓ CORRECT" if answer_em == 1.0 else "\n✗ WRONG")
        self._rule()

    def summary(self, records: list[dict[str, Any]], metrics: dict[str, float], config: dict[str, Any], artifacts: dict[str, str], elapsed: float) -> None:
        if not self.enabled:
            return
        count = len(records); completed = sum(not row.get("runtime_error") for row in records)
        passed = sum(not row.get("runtime_error") and score_hotpot_record(row).get("em") == 1.0 for row in records)
        average = lambda key: sum(float(row.get(key, 0) or 0) for row in records) / count if count else 0.0
        self._rule(); print("HotpotQA Aggregate Summary"); self._rule()
        print("Run completion")
        for label, value in (("Selected samples", config.get("sample_count", count)), ("Completed samples", completed), ("Scored samples", metrics.get("scored", 0)), ("Questions passed", f"{passed}/{int(metrics.get('scored', 0))}"), ("Runtime failures", count - completed), ("Elapsed run time", f"{elapsed:.2f} seconds"), ("Average latency", f"{average('total_latency'):.2f} seconds")):
            self._field(label, value)
        for heading, prefix in (("Answer metrics", ""), ("Supporting-fact metrics", "sp_"), ("Joint metrics", "joint_")):
            print(f"\n{heading}")
            for label, suffix in (("EM", "em"), ("F1", "f1"), ("Precision", "prec"), ("Recall", "recall")):
                self._field(label, f"{metrics.get(prefix + suffix, 0.0):.2%}")
        print("\nReasoning diagnostics")
        for label, key in (("Average hops", "reasoning_hops"), ("Average retrievals", "retrieval_calls"), ("Average reflections", "reflection_interventions"), ("Average evidence", "evidence_count")):
            self._field(label, f"{average(key):.2f}")
        self._field("Stop reasons", dict(Counter(str(row.get("final_stop_reason") or "none") for row in records)))
        self._field("Failure categories", dict(Counter(str(row.get("failure_category")) for row in records if row.get("failure_category"))))
        print("\nRun identity")
        for label, key in (("Dataset set", "dataset_set"), ("Mode", "mode"), ("Provider", "provider"), ("Model", "model"), ("Resolved seed", "resolved_seed")):
            self._field(label, config.get(key, "-"))
        print("\nArtifacts")
        for label, key in (("Predictions", "predictions"), ("Metrics", "metrics"), ("Manifest", "manifest"), ("Checkpoint", "checkpoint"), ("Log directory", "logs")):
            self._field(label, artifacts.get(key, "-"))
        self._rule()
