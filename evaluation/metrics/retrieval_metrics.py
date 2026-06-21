"""Standard information retrieval metrics."""

from __future__ import annotations

import math
from typing import Any


def recall_at_k(expected_ids: list[str] | tuple[str, ...], retrieved_ids: list[str] | tuple[str, ...], k: int) -> float:
    """Compute Recall@K for one query."""
    expected = set(expected_ids)
    if not expected:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    return round(len(expected & retrieved) / len(expected), 6)


def reciprocal_rank(expected_ids: list[str] | tuple[str, ...], retrieved_ids: list[str] | tuple[str, ...]) -> float:
    """Compute reciprocal rank for one query."""
    expected = set(expected_ids)
    for index, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in expected:
            return round(1.0 / index, 6)
    return 0.0


def dcg_at_k(expected_ids: list[str] | tuple[str, ...], retrieved_ids: list[str] | tuple[str, ...], k: int) -> float:
    """Compute binary relevance DCG@K."""
    expected = set(expected_ids)
    total = 0.0
    for index, retrieved_id in enumerate(retrieved_ids[:k], start=1):
        relevance = 1.0 if retrieved_id in expected else 0.0
        total += relevance / math.log2(index + 1)
    return total


def ndcg_at_k(expected_ids: list[str] | tuple[str, ...], retrieved_ids: list[str] | tuple[str, ...], k: int) -> float:
    """Compute binary relevance nDCG@K."""
    ideal_hits = min(len(expected_ids), k)
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    if ideal == 0.0:
        return 0.0
    return round(dcg_at_k(expected_ids, retrieved_ids, k) / ideal, 6)


def hit_rate(expected_ids: list[str] | tuple[str, ...], retrieved_ids: list[str] | tuple[str, ...]) -> float:
    """Return 1.0 when any expected memory was retrieved, else 0.0."""
    return 1.0 if set(expected_ids) & set(retrieved_ids) else 0.0


def summarize_retrieval_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate standard retrieval metrics over benchmark trace records."""
    if not records:
        return {
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "ndcg_at_5": 0.0,
            "hit_rate": 0.0,
            "average_retrieval_latency": 0.0,
            "average_retrieved_memory_count": 0.0,
        }
    total = len(records)
    return {
        "recall_at_1": _average(records, lambda record: recall_at_k(record["expected_memory_ids"], record["retrieved_memory_ids"], 1)),
        "recall_at_3": _average(records, lambda record: recall_at_k(record["expected_memory_ids"], record["retrieved_memory_ids"], 3)),
        "recall_at_5": _average(records, lambda record: recall_at_k(record["expected_memory_ids"], record["retrieved_memory_ids"], 5)),
        "mrr": _average(records, lambda record: reciprocal_rank(record["expected_memory_ids"], record["retrieved_memory_ids"])),
        "ndcg_at_5": _average(records, lambda record: ndcg_at_k(record["expected_memory_ids"], record["retrieved_memory_ids"], 5)),
        "hit_rate": _average(records, lambda record: hit_rate(record["expected_memory_ids"], record["retrieved_memory_ids"])),
        "average_retrieval_latency": round(sum(float(record.get("latency_ms", 0.0)) for record in records) / total, 6),
        "average_retrieved_memory_count": round(
            sum(len(record.get("retrieved_memory_ids", ())) for record in records) / total,
            6,
        ),
    }


def _average(records: list[dict[str, Any]], fn: Any) -> float:
    return round(sum(float(fn(record)) for record in records) / len(records), 6)
