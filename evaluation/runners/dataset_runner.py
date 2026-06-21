"""Dataset replay runner for PRIMA-NEXT runtime validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from affect import DynamicAffectEngine
from evaluation.metrics.affect_metrics import emotion_frequency, pad_distributions
from evaluation.metrics.confidence_calibration import (
    confidence_calibration_summary,
    print_confidence_calibration_report,
)
from evaluation.metrics.memory_metrics import memory_growth, retrieval_hit_counts
from evaluation.metrics.reflection_accuracy import print_reflection_accuracy_report, reflection_accuracy_summary
from evaluation.metrics.reflection_harm import reflection_harm_summary
from evaluation.metrics.reflection_metrics import (
    affect_trigger_count,
    correction_frequency,
    correction_rate,
    contradiction_trigger_count,
    reflection_rate,
    retrieval_trigger_count,
    trigger_distribution,
    trigger_frequency,
    trigger_summary,
    utility_summary,
)
from evaluation.metrics.system_metrics import summarize_system_metrics
from evaluation.runners.emotion_eval_runner import canonical_label
from reflection.reflection_acceptance import (
    ReflectionAcceptanceDecision,
    classify_reflection_outcome,
    evaluate_reflection_acceptance,
    reflection_recovery_summary,
)
from reflection.reflection_context import ReflectionContext
from reflection.reflection_engine import ReflectionEngine
from runtime.prima_runtime import PrimaRuntime


DEFAULT_DATASET_PATH = Path("evaluation/datasets/inputs_100.json")
DEFAULT_RESULTS_PATH = Path("evaluation/results/runtime_results.json")
DEFAULT_METRICS_PATH = Path("evaluation/results/runtime_metrics.json")
DEFAULT_REFLECTION_LOG_PATH = Path("evaluation/results/reflection_log.json")
DEFAULT_REFLECTION_CHANGE_REPORT_PATH = Path("evaluation/results/reflection_change_report.json")
DEFAULT_REFLECTION_ACCURACY_PATH = Path("evaluation/results/reflection_accuracy.json")
DEFAULT_CONFIDENCE_CALIBRATION_PATH = Path("evaluation/results/confidence_calibration.json")
DEFAULT_REFLECTION_GOLD_RESULTS_PATH = Path("evaluation/results/reflection_gold_results.json")
DEFAULT_REFLECTION_OUTCOMES_PATH = Path("evaluation/results/reflection_outcomes.json")
DEFAULT_REFLECTION_RECOVERY_REPORT_PATH = Path("evaluation/results/reflection_recovery_report.json")


class DatasetRunner:
    """Replay utterance datasets through PrimaRuntime and persist results."""

    def __init__(
        self,
        runtime: PrimaRuntime | None = None,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        results_path: str | Path = DEFAULT_RESULTS_PATH,
        metrics_path: str | Path = DEFAULT_METRICS_PATH,
        reflection_log_path: str | Path = DEFAULT_REFLECTION_LOG_PATH,
        reflection_change_report_path: str | Path = DEFAULT_REFLECTION_CHANGE_REPORT_PATH,
        reflection_accuracy_path: str | Path = DEFAULT_REFLECTION_ACCURACY_PATH,
        confidence_calibration_path: str | Path = DEFAULT_CONFIDENCE_CALIBRATION_PATH,
        reflection_gold_results_path: str | Path = DEFAULT_REFLECTION_GOLD_RESULTS_PATH,
        reflection_outcomes_path: str | Path = DEFAULT_REFLECTION_OUTCOMES_PATH,
        reflection_recovery_report_path: str | Path = DEFAULT_REFLECTION_RECOVERY_REPORT_PATH,
    ) -> None:
        self.runtime = runtime or PrimaRuntime()
        self.dataset_path = Path(dataset_path)
        self.results_path = Path(results_path)
        self.metrics_path = Path(metrics_path)
        self.reflection_log_path = Path(reflection_log_path)
        self.reflection_change_report_path = Path(reflection_change_report_path)
        self.reflection_accuracy_path = Path(reflection_accuracy_path)
        self.confidence_calibration_path = Path(confidence_calibration_path)
        self.reflection_gold_results_path = Path(reflection_gold_results_path)
        self.reflection_outcomes_path = Path(reflection_outcomes_path)
        self.reflection_recovery_report_path = Path(reflection_recovery_report_path)
        self._affect_engine = DynamicAffectEngine()
        self._reflection_engine = ReflectionEngine()

    def run(self) -> list[dict[str, Any]]:
        """Process every dataset sample and append compact result records."""
        samples = self._load_samples()
        if samples and all("ground_truth_emotion" in sample for sample in samples):
            records = self._run_reflection_gold(samples)
            self._write_reflection_gold_results(records)
            self._write_reflection_outcomes(records)
            self._write_reflection_recovery_report(records)
            self._write_reflection_accuracy(records)
            self._write_metrics(records)
            return records

        records = []
        for sample in samples:
            query = str(sample.get("query") or sample.get("raw_text") or "")
            try:
                result = self.runtime.process(query)
                affect_state = result.affect_state
                record = {
                    "query": query,
                    "emotion": str(affect_state.get("dominant_emotion", "neutral")),
                    "confidence": result.confidence_score,
                    "reflection_triggered": result.reflection_triggered,
                    "prediction_before_reflection": result.prediction_before_reflection,
                    "prediction_after_reflection": result.prediction_after_reflection,
                    "reflection_reasons": [dict(reason) for reason in result.reflection_reasons],
                    "reflection_before_confidence": result.reflection_before_confidence,
                    "reflection_after_confidence": result.reflection_after_confidence,
                    "reflection_utility_score": result.reflection_utility_score,
                    "correction_count": result.correction_count,
                    "memory_created": len(result.memory_notes_created) > 0,
                    "retrieval_count": len(result.retrieved_memories),
                    "latency_ms": result.latency_ms,
                    "valence": float(affect_state.get("valence", 0.0)),
                    "arousal": float(affect_state.get("arousal", 0.0)),
                    "dominance": float(affect_state.get("dominance", 0.0)),
                    "errors": list(result.errors),
                }
                if "ground_truth" in sample:
                    record["ground_truth"] = str(sample.get("ground_truth", ""))
                if "prediction_before_reflection" in sample:
                    record["prediction_before_reflection"] = str(sample.get("prediction_before_reflection", ""))
                if "prediction_after_reflection" in sample:
                    record["prediction_after_reflection"] = str(sample.get("prediction_after_reflection", ""))
                records.append(record)
            except Exception as exc:
                records.append(
                    {
                        "query": query,
                        "emotion": "error",
                        "confidence": 0.0,
                        "reflection_triggered": False,
                        "prediction_before_reflection": "",
                        "prediction_after_reflection": "",
                        "reflection_reasons": [],
                        "reflection_before_confidence": 0.0,
                        "reflection_after_confidence": 0.0,
                        "reflection_utility_score": 0.0,
                        "correction_count": 0,
                        "memory_created": False,
                        "retrieval_count": 0,
                        "latency_ms": 0.0,
                        "errors": [str(exc)],
                    }
                )
        self._append_results(records)
        self._write_reflection_log(records)
        self._write_reflection_change_report(records)
        self._write_reflection_accuracy(records)
        self._write_confidence_calibration(records)
        self._write_metrics(records)
        return records

    def _run_reflection_gold(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for sample in samples:
            query = str(sample.get("query", ""))
            ground_truth = canonical_label(str(sample.get("ground_truth_emotion", "")))
            profile = self._affect_engine.get_emotion_profile(query)
            before_prediction = canonical_label(profile.dominant_emotion)
            decision = self._reflection_decision(query, before_prediction, profile.confidence, profile.emotions)
            after_prediction = decision.selected_prediction
            outcome = classify_reflection_outcome(
                before_prediction,
                after_prediction,
                ground_truth,
                reflection_triggered=decision.reflection_triggered,
            )
            records.append(
                {
                    "query": query,
                    "ground_truth": ground_truth,
                    "before_prediction": before_prediction,
                    "after_prediction": after_prediction,
                    "reflection_candidate_prediction": decision.candidate_prediction,
                    "reflection_triggered": decision.reflection_triggered,
                    "reflection_accepted": decision.reflection_accepted,
                    "reflection_outcome": outcome,
                    "reflection_suppression_reason": decision.suppression_reason,
                    "reflection_acceptance_reason": decision.acceptance_reason,
                    "reflection_before_confidence": decision.original_confidence,
                    "reflection_after_confidence": decision.candidate_confidence,
                    "original_evidence_score": decision.original_evidence_score,
                    "reflection_evidence_score": decision.reflection_evidence_score,
                    "top_emotion_margin": decision.top_emotion_margin,
                    "correct_before": before_prediction == ground_truth,
                    "correct_after": after_prediction == ground_truth,
                    "confidence": float(profile.confidence),
                }
            )
        return records

    def _write_reflection_log(self, records: list[dict[str, Any]]) -> None:
        self.reflection_log_path.parent.mkdir(parents=True, exist_ok=True)
        reflection_records = [
            {
                "query": record.get("query", ""),
                "before_prediction": str(record.get("prediction_before_reflection", "")),
                "after_prediction": str(record.get("prediction_after_reflection", "")),
                "before_confidence": float(record.get("reflection_before_confidence", 0.0)),
                "after_confidence": float(record.get("reflection_after_confidence", 0.0)),
                "changed": str(record.get("prediction_before_reflection", "")) != str(record.get("prediction_after_reflection", "")),
            }
            for record in records
        ]
        self.reflection_log_path.write_text(json.dumps(reflection_records, indent=2), encoding="utf-8")

    def _write_reflection_change_report(self, records: list[dict[str, Any]]) -> None:
        self.reflection_change_report_path.parent.mkdir(parents=True, exist_ok=True)
        changed_records = [
            {
                "query": str(record.get("query", "")),
                "before_prediction": str(record.get("prediction_before_reflection", "")),
                "after_prediction": str(record.get("prediction_after_reflection", "")),
                "before_confidence": float(record.get("reflection_before_confidence", 0.0)),
                "after_confidence": float(record.get("reflection_after_confidence", 0.0)),
                "changed": True,
            }
            for record in records
            if str(record.get("prediction_before_reflection", "")) != str(record.get("prediction_after_reflection", ""))
        ]
        self.reflection_change_report_path.write_text(json.dumps(changed_records, indent=2), encoding="utf-8")

    def _load_samples(self) -> list[dict[str, Any]]:
        raw = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        return []

    def _append_results(self, records: list[dict[str, Any]]) -> None:
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict[str, Any]] = []
        if self.results_path.exists():
            existing_payload = json.loads(self.results_path.read_text(encoding="utf-8"))
            if isinstance(existing_payload, list):
                existing = existing_payload
        self.results_path.write_text(json.dumps(existing + records, indent=2), encoding="utf-8")

    def _write_metrics(self, records: list[dict[str, Any]]) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        reflection_accuracy = reflection_accuracy_summary(records)
        reflection_harm = reflection_harm_summary(records)
        confidence_calibration = confidence_calibration_summary(records)
        payload = {
            "system": summarize_system_metrics(records),
            "affect": {
                "emotion_frequency": emotion_frequency(records),
                "pad_distributions": pad_distributions(records),
            },
            "memory": {
                "retrieval_hit_counts": retrieval_hit_counts(records),
                "memory_growth": memory_growth(records),
            },
            "reflection": {
                "trigger_frequency": trigger_frequency(records),
                "reflection_rate": reflection_rate(records),
                "correction_frequency": correction_frequency(records),
                "correction_rate": correction_rate(records),
                "affect_trigger_count": affect_trigger_count(records),
                "retrieval_trigger_count": retrieval_trigger_count(records),
                "contradiction_trigger_count": contradiction_trigger_count(records),
                "trigger_distribution": trigger_distribution(records),
                "trigger_summary": trigger_summary(records),
                "utility_summary": utility_summary(records),
                "success_criteria": {
                    "reflection_rate_min": 0.05,
                    "reflection_rate_max": 0.15,
                    "correction_rate_min": 0.15,
                    "correction_rate_max": 0.25,
                },
            },
            "reflection_accuracy": reflection_accuracy,
            "reflection_harm": reflection_harm,
            "reflection_recovery": reflection_recovery_summary(records),
            "confidence_calibration": confidence_calibration,
        }
        self.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print_reflection_accuracy_report(records)
        print_confidence_calibration_report(records)

    def _write_reflection_accuracy(self, records: list[dict[str, Any]]) -> None:
        self.reflection_accuracy_path.parent.mkdir(parents=True, exist_ok=True)
        self.reflection_accuracy_path.write_text(
            json.dumps(reflection_accuracy_summary(records), indent=2),
            encoding="utf-8",
        )

    def _write_confidence_calibration(self, records: list[dict[str, Any]]) -> None:
        self.confidence_calibration_path.parent.mkdir(parents=True, exist_ok=True)
        self.confidence_calibration_path.write_text(
            json.dumps(confidence_calibration_summary(records), indent=2),
            encoding="utf-8",
        )

    def _write_reflection_gold_results(self, records: list[dict[str, Any]]) -> None:
        self.reflection_gold_results_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "query": str(record.get("query", "")),
                "ground_truth": str(record.get("ground_truth", "")),
                "before_prediction": str(record.get("before_prediction", "")),
                "after_prediction": str(record.get("after_prediction", "")),
                "reflection_candidate_prediction": str(record.get("reflection_candidate_prediction", "")),
                "reflection_triggered": bool(record.get("reflection_triggered", False)),
                "reflection_accepted": bool(record.get("reflection_accepted", False)),
                "reflection_outcome": str(record.get("reflection_outcome", "NEUTRAL")),
                "correct_before": bool(record.get("correct_before", False)),
                "correct_after": bool(record.get("correct_after", False)),
            }
            for record in records
        ]
        self.reflection_gold_results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_reflection_outcomes(self, records: list[dict[str, Any]]) -> None:
        self.reflection_outcomes_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "query": str(record.get("query", "")),
                "ground_truth": str(record.get("ground_truth", "")),
                "before_prediction": str(record.get("before_prediction", "")),
                "reflection_candidate_prediction": str(record.get("reflection_candidate_prediction", "")),
                "after_prediction": str(record.get("after_prediction", "")),
                "reflection_triggered": bool(record.get("reflection_triggered", False)),
                "reflection_accepted": bool(record.get("reflection_accepted", False)),
                "outcome": str(record.get("reflection_outcome", "NEUTRAL")),
                "original_evidence_score": float(record.get("original_evidence_score", 0.0)),
                "reflection_evidence_score": float(record.get("reflection_evidence_score", 0.0)),
                "top_emotion_margin": float(record.get("top_emotion_margin", 0.0)),
            }
            for record in records
            if record.get("reflection_triggered")
        ]
        self.reflection_outcomes_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_reflection_recovery_report(self, records: list[dict[str, Any]]) -> None:
        self.reflection_recovery_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.reflection_recovery_report_path.write_text(
            json.dumps(reflection_recovery_summary(records), indent=2),
            encoding="utf-8",
        )

    def _reflection_decision(
        self,
        query: str,
        before_prediction: str,
        affect_confidence: float,
        emotions: dict[str, float],
    ) -> ReflectionAcceptanceDecision:
        reflection_context = ReflectionContext(
            query=query,
            affect_confidence=affect_confidence,
            failure_metadata={
                "reason": "low confidence emotion benchmark",
                "severity": round(max(0.0, 1.0 - affect_confidence), 6),
                "threshold": 0.35,
                "audit_sample": True,
            },
        )
        result = self._reflection_engine.evaluate(reflection_context)
        ranked_labels = _unique_labels(canonical_label(label) for label in emotions.keys())
        original = before_prediction or (ranked_labels[0] if ranked_labels else "joy")
        candidate = ranked_labels[1] if result.should_reflect and len(ranked_labels) > 1 else original
        return evaluate_reflection_acceptance(
            query=query,
            original_prediction=original,
            candidate_prediction=candidate,
            prediction_confidence=affect_confidence,
            emotions=emotions,
            original_confidence=result.before_confidence,
            candidate_confidence=result.after_confidence,
        )


def _unique_labels(labels: Any) -> list[str]:
    unique: list[str] = []
    for label in labels:
        if label not in unique:
            unique.append(label)
    return unique


def run_dataset() -> list[dict[str, Any]]:
    """Run the default dataset replay."""
    return DatasetRunner().run()


if __name__ == "__main__":
    run_dataset()
