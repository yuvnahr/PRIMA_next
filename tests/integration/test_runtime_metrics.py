"""Runtime metrics tests."""

from __future__ import annotations

from evaluation.metrics.affect_metrics import emotion_frequency
from evaluation.metrics.memory_metrics import memory_growth, retrieval_hit_counts
from evaluation.metrics.reflection_metrics import trigger_frequency
from evaluation.metrics.system_metrics import summarize_system_metrics
from llm.llm_types import LLMResponse
from runtime import PrimaRuntime
from runtime.runtime_metrics import RuntimeMetrics


class _ModelClient:
    provider_name = "test"

    def chat(self, **_kwargs: object) -> LLMResponse:
        return LLMResponse("Generated metrics response.")


def test_runtime_metrics_are_serializable() -> None:
    runtime = PrimaRuntime(llm_client=_ModelClient())
    result = runtime.process("A calm morning in the garden helped me focus.")
    metrics = runtime.metrics_from_result(result)

    assert isinstance(metrics, RuntimeMetrics)
    payload = metrics.to_dict()
    assert payload["latency_ms"] >= 0.0
    assert payload["emotion_classification"]


def test_metric_helpers_summarize_records() -> None:
    records = [
        {
            "emotion": "joy",
            "confidence": 0.8,
            "reflection_triggered": True,
            "memory_created": True,
            "retrieval_count": 2,
            "latency_ms": 10.0,
        },
        {
            "emotion": "fear",
            "confidence": 0.4,
            "reflection_triggered": False,
            "memory_created": True,
            "retrieval_count": 1,
            "latency_ms": 30.0,
        },
    ]

    assert summarize_system_metrics(records)["average_latency"] == 20.0
    assert emotion_frequency(records) == {"joy": 1, "fear": 1}
    assert retrieval_hit_counts(records) == [2, 1]
    assert memory_growth(records) == 2
    assert trigger_frequency(records) == 1
