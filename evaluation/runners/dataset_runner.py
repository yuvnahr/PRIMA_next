"""Dataset replay runner for PRIMA-NEXT runtime validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.metrics.affect_metrics import emotion_frequency, pad_distributions
from evaluation.metrics.memory_metrics import memory_growth, retrieval_hit_counts
from evaluation.metrics.reflection_metrics import (
    correction_frequency,
    correction_rate,
    reflection_rate,
    trigger_distribution,
    trigger_frequency,
    utility_summary,
)
from evaluation.metrics.system_metrics import summarize_system_metrics
from runtime.prima_runtime import PrimaRuntime


DEFAULT_DATASET_PATH = Path("evaluation/datasets/inputs_100.json")
DEFAULT_RESULTS_PATH = Path("evaluation/results/runtime_results.json")
DEFAULT_METRICS_PATH = Path("evaluation/results/runtime_metrics.json")
DEFAULT_REFLECTION_LOG_PATH = Path("evaluation/results/reflection_log.json")


class DatasetRunner:
    """Replay utterance datasets through PrimaRuntime and persist results."""

    def __init__(
        self,
        runtime: PrimaRuntime | None = None,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        results_path: str | Path = DEFAULT_RESULTS_PATH,
        metrics_path: str | Path = DEFAULT_METRICS_PATH,
        reflection_log_path: str | Path = DEFAULT_REFLECTION_LOG_PATH,
    ) -> None:
        self.runtime = runtime or PrimaRuntime()
        self.dataset_path = Path(dataset_path)
        self.results_path = Path(results_path)
        self.metrics_path = Path(metrics_path)
        self.reflection_log_path = Path(reflection_log_path)

    def run(self) -> list[dict[str, Any]]:
        """Process every dataset sample and append compact result records."""
        records: list[dict[str, Any]] = []
        for sample in self._load_samples():
            query = str(sample.get("raw_text", ""))
            try:
                result = self.runtime.process(query)
                affect_state = result.affect_state
                records.append(
                    {
                        "query": query,
                        "emotion": str(affect_state.get("dominant_emotion", "neutral")),
                        "confidence": result.confidence_score,
                        "reflection_triggered": result.reflection_triggered,
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
                )
            except Exception as exc:
                records.append(
                    {
                        "query": query,
                        "emotion": "error",
                        "confidence": 0.0,
                        "reflection_triggered": False,
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
        self._write_metrics(records)
        return records

    def _write_reflection_log(self, records: list[dict[str, Any]]) -> None:
        self.reflection_log_path.parent.mkdir(parents=True, exist_ok=True)
        reflection_records = [
            {
                "query": record.get("query", ""),
                "reflection_triggered": bool(record.get("reflection_triggered", False)),
                "reasons": list(record.get("reflection_reasons", [])),
                "before_confidence": float(record.get("reflection_before_confidence", 0.0)),
                "after_confidence": float(record.get("reflection_after_confidence", 0.0)),
                "utility_score": float(record.get("reflection_utility_score", 0.0)),
                "correction_count": int(record.get("correction_count", 0)),
            }
            for record in records
        ]
        self.reflection_log_path.write_text(json.dumps(reflection_records, indent=2), encoding="utf-8")

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
                "trigger_distribution": trigger_distribution(records),
                "utility_summary": utility_summary(records),
                "success_criteria": {
                    "reflection_rate_min": 0.10,
                    "reflection_rate_max": 0.20,
                    "correction_rate_min": 0.15,
                    "correction_rate_max": 0.25,
                },
            },
        }
        self.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_dataset() -> list[dict[str, Any]]:
    """Run the default dataset replay."""
    return DatasetRunner().run()


if __name__ == "__main__":
    run_dataset()
