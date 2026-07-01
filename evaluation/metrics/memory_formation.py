"""Metrics for adaptive memory formation."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository, MemoryRepository
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_controller import RetrievalController
from memory.retrieval.retrieval_request import RetrievalRequest


def memory_formation_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(records)
    stored = sum(1 for record in records if record.get("memory_created"))
    return {
        "total_interactions": total,
        "stored_memories": stored,
        "rejected_memories": total - stored,
        "memory_creation_rate": _rate(stored, total),
        "memory_rejection_rate": _rate(total - stored, total),
        "average_importance_score": _average(records, "memory_importance_total"),
        "average_novelty": _average(records, "memory_importance_novelty"),
        "average_salience": _average(records, "memory_importance_salience"),
        "importance_retention_correlation": _importance_retention_correlation(records),
    }


def memory_quality_comparison(records: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
    benchmark_records = [
        record
        for record in records
        if record.get("memory_created") or float(record.get("memory_importance_total", 0.0)) >= float(record.get("memory_importance_threshold", 0.55))
    ]
    before_repository = _repository_from_records(records)
    after_repository = _repository_from_records([record for record in records if record.get("memory_created")])
    before = _retrieval_quality(before_repository, benchmark_records, top_k)
    after = _retrieval_quality(after_repository, benchmark_records, top_k)
    return {
        "benchmark_queries": len(benchmark_records),
        "top_k": top_k,
        "before_memory_filtering": before,
        "after_memory_filtering": after,
        "recall_drop": round(float(before["recall_at_k"]) - float(after["recall_at_k"]), 6),
        "hit_rate_drop": round(float(before["hit_rate"]) - float(after["hit_rate"]), 6),
        "mrr_drop": round(float(before["mrr"]) - float(after["mrr"]), 6),
    }


def _retrieval_quality(repository: MemoryRepository, records: list[dict[str, Any]], top_k: int) -> dict[str, float]:
    if not records:
        return {"recall_at_k": 0.0, "mrr": 0.0, "hit_rate": 0.0}
    controller = RetrievalController(repository)
    hits = 0
    reciprocal_rank_total = 0.0
    for record in records:
        expected_id = _note_id(record)
        response = controller.retrieve(RetrievalRequest(query=str(record.get("query", "")), top_k=top_k))
        result_ids = [result.note.id for result in response.results]
        if expected_id in result_ids:
            hits += 1
            reciprocal_rank_total += 1.0 / (result_ids.index(expected_id) + 1)
    return {
        "recall_at_k": _rate(hits, len(records)),
        "mrr": round(reciprocal_rank_total / len(records), 6),
        "hit_rate": _rate(hits, len(records)),
    }


def _repository_from_records(records: list[dict[str, Any]]) -> InMemoryMemoryRepository:
    repository = InMemoryMemoryRepository()
    for record in records:
        query = str(record.get("query", ""))
        if not query:
            continue
        repository.add(MemoryNote.create(query, memory_type=MemoryType.EPISODIC, note_id=_note_id(record)))
    return repository


def _note_id(record: dict[str, Any]) -> str:
    source = str(record.get("query", ""))
    return f"quality_{hashlib.sha256(source.encode('utf-8')).hexdigest()[:16]}"


def _average(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record.get(key, 0.0)) for record in records]
    return round(sum(values) / len(values), 6) if values else 0.0


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _importance_retention_correlation(records: list[dict[str, Any]]) -> float:
    if len(records) < 2:
        return 0.0
    scores = [float(record.get("memory_importance_total", 0.0)) for record in records]
    retained = [1.0 if record.get("memory_created") else 0.0 for record in records]
    mean_score = sum(scores) / len(scores)
    mean_retained = sum(retained) / len(retained)
    numerator = sum((score - mean_score) * (stored - mean_retained) for score, stored in zip(scores, retained))
    score_variance = sum((score - mean_score) ** 2 for score in scores)
    retained_variance = sum((stored - mean_retained) ** 2 for stored in retained)
    denominator = math.sqrt(score_variance * retained_variance)
    return round(numerator / denominator, 6) if denominator else 0.0
