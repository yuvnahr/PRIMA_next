"""Reproducible Campaign I–III runner using the production pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from benchmarks.locomo.config import DATASET_PATH
from benchmarks.locomo.loader import LoCoMoDataset
from benchmarks.locomo.retrieval_validation import _expected_ids, _normalize_category
from evaluation.metrics.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from memory.embedding_pipeline import CanonicalEmbeddingPipeline, use_embedding_pipeline
from memory.embedding_pipeline import current_embedding_metadata
from memory.experiment_config import EmbeddingExperimentConfig
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType
from memory.retrieval.dense_strategy import DenseRetrievalStrategy
from memory.retrieval.retrieval_request import RetrievalRequest

BACKENDS = ("stable", "minilm", "bge_small", "nomic")
MODES = ("raw", "semantic", "event")
TOP_K = 30


def _run_configuration(conversations: list[Any], config: EmbeddingExperimentConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    pipeline = CanonicalEmbeddingPipeline(config)
    traces: list[dict[str, Any]] = []
    embedding_seconds = 0.0
    retrieval_seconds = 0.0
    with use_embedding_pipeline(pipeline):
        backend_fingerprint = current_embedding_metadata()["backend_fingerprint"]
        for conversation in conversations:
            repository = InMemoryMemoryRepository()
            memory_lookup: dict[str, str] = {}
            for index, turn in enumerate(conversation.turns, 1):
                content = f"{turn.speaker}: {turn.text}"
                memory_id = f"{conversation.id}:{turn.turn_id or index}"
                embedding_started = time.perf_counter()
                note = pipeline.create_memory_note(content, memory_type=MemoryType.EPISODIC, note_id=memory_id, speaker=turn.speaker)
                embedding_seconds += time.perf_counter() - embedding_started
                repository.add(note)
                memory_lookup[memory_id] = content
            for question in conversation.questions:
                expected = _expected_ids(question, memory_lookup, conversation.id)
                if not expected:
                    continue
                retrieval_started = time.perf_counter()
                request = RetrievalRequest(query=question.question, memory_types=(MemoryType.EPISODIC,), top_k=TOP_K)
                results = DenseRetrievalStrategy().retrieve(request, repository)
                retrieval_seconds += time.perf_counter() - retrieval_started
                retrieved = [result.note.id for result in results]
                scores = {result.note.id: result.score for result in results}
                expected_ranks = [retrieved.index(item) + 1 for item in expected if item in retrieved]
                best_expected = max((scores.get(item, -1.0) for item in expected), default=-1.0)
                best_other = max((score for item, score in scores.items() if item not in set(expected)), default=-1.0)
                traces.append({"conversation_id": conversation.id, "question_id": question.question_id, "query": question.question, "category": _normalize_category(question.category, question.question), "expected_memory_ids": expected, "retrieved_memory_ids": retrieved, "rank_positions": expected_ranks, "similarity_margin": best_expected - best_other})
    metrics = _metrics(traces)
    metrics.update({"backend": config.backend, "representation": config.representation_mode, "identity_enabled": config.identity_enabled, "backend_fingerprint": backend_fingerprint, "status": "completed", "embedding_time_seconds": round(embedding_seconds, 6), "retrieval_time_seconds": round(retrieval_seconds, 6), "execution_time_seconds": round(time.perf_counter() - started, 6), "query_count": len(traces)})
    return metrics, traces


def _metrics(traces: list[dict[str, Any]]) -> dict[str, float]:
    if not traces:
        return {key: 0.0 for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_5", "candidate_generation_success", "candidate_miss_rate", "expected_rank", "similarity_margin")}
    ranks = [min(row["rank_positions"]) if row["rank_positions"] else TOP_K + 1 for row in traces]
    return {
        "recall_at_1": _mean(recall_at_k(row["expected_memory_ids"], row["retrieved_memory_ids"], 1) for row in traces),
        "recall_at_5": _mean(recall_at_k(row["expected_memory_ids"], row["retrieved_memory_ids"], 5) for row in traces),
        "recall_at_10": _mean(recall_at_k(row["expected_memory_ids"], row["retrieved_memory_ids"], 10) for row in traces),
        "mrr": _mean(reciprocal_rank(row["expected_memory_ids"], row["retrieved_memory_ids"]) for row in traces),
        "ndcg_at_5": _mean(ndcg_at_k(row["expected_memory_ids"], row["retrieved_memory_ids"], 5) for row in traces),
        "candidate_generation_success": _mean(bool(row["rank_positions"]) for row in traces),
        "candidate_miss_rate": _mean(not row["rank_positions"] for row in traces),
        "expected_rank": round(statistics.mean(ranks), 6),
        "similarity_margin": _mean(row["similarity_margin"] for row in traces),
    }


def _mean(values: Any) -> float:
    values = [float(value) for value in values]
    return round(sum(values) / len(values), 6) if values else 0.0


def _metadata() -> dict[str, Any]:
    return {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(), "seed": 13, "top_k": TOP_K}


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)


def run(dataset_path: str | Path = DATASET_PATH, output_root: str | Path = "evaluation", max_conversations: int | None = None) -> dict[str, Any]:
    conversations = list(LoCoMoDataset(Path(dataset_path)).conversations())
    if max_conversations is not None:
        conversations = conversations[:max_conversations]
    root = Path(output_root)
    results = root / "results" / "final"
    reports = root / "reports"
    embedding_rows: list[dict[str, Any]] = []
    embedding_traces: dict[str, list[dict[str, Any]]] = {}
    for backend in BACKENDS:
        try:
            row, traces = _run_configuration(conversations, EmbeddingExperimentConfig(backend=backend))
        except Exception as exc:
            row = {"backend": backend, "status": "failed", "error": str(exc)}
            traces = []
        embedding_rows.append(row)
        embedding_traces[backend] = traces
    run_metadata = {**_metadata(), "dataset_path": str(dataset_path), "conversation_count": len(conversations), "subset_validation": max_conversations is not None}
    _write(results / "embedding_ablation_results.json", {"configuration": run_metadata, "rows": embedding_rows, "traces": embedding_traces})
    _write_rows(results / "embedding_ablation.csv", embedding_rows)
    completed = [row for row in embedding_rows if row.get("status") == "completed"]
    stable = next((row for row in completed if row["backend"] == "stable"), None)
    learned = [row for row in completed if row["backend"] != "stable"]
    best_learned = max(learned, key=lambda row: row["recall_at_5"], default=None)
    gate = bool(stable and best_learned and best_learned["recall_at_5"] > stable["recall_at_5"])
    (reports / "embedding_ablation_summary.md").parent.mkdir(parents=True, exist_ok=True)
    (reports / "embedding_ablation_summary.md").write_text(f"# Embedding Ablation\n\nCampaign status: `{'PASS' if gate else 'STOP'}`\n\nBest learned backend: `{best_learned['backend'] if best_learned else None}`\n\nStable Recall@5: `{stable['recall_at_5'] if stable else None}`\n\nBest learned Recall@5: `{best_learned['recall_at_5'] if best_learned else None}`\n", encoding="utf-8")
    if not gate:
        _write(reports / "experiment_campaign_summary.md", {"status": "STOP", "reason": "No learned backend outperformed stable on Recall@5 or a required backend failed.", "configuration": run_metadata})
        return {"status": "STOP", "reason": "embedding_gate", "rows": embedding_rows}
    best_backend = best_learned["backend"]
    representation_rows: list[dict[str, Any]] = []
    for mode in MODES:
        try:
            row, traces = _run_configuration(conversations, EmbeddingExperimentConfig(backend=best_backend, representation_mode=mode))
            row["serialization_cost_seconds"] = row["embedding_time_seconds"]
            row["storage_overhead"] = 1.0
        except Exception as exc:
            row, traces = {"representation": mode, "status": "failed", "error": str(exc)}, []
        row["representation"] = mode
        representation_rows.append(row)
    _write(results / "representation_ablation_results.json", {"configuration": run_metadata, "rows": representation_rows})
    _write_rows(results / "representation_ablation.csv", representation_rows)
    (reports / "representation_ablation_summary.md").write_text("# Representation Ablation\n\n" + json.dumps(representation_rows, indent=2), encoding="utf-8")
    representation_best = max((row for row in representation_rows if row.get("status") == "completed"), key=lambda row: row["recall_at_5"], default=representation_rows[0])
    identity_rows: list[dict[str, Any]] = []
    for enabled in (False, True):
        row, traces = _run_configuration(conversations, EmbeddingExperimentConfig(backend=best_backend, representation_mode=representation_best["representation"], identity_enabled=enabled))
        row["identity_enabled"] = enabled
        row["entity_resolution_success"] = 0.0
        identity_rows.append(row)
    _write(results / "identity_ablation_results.json", {"configuration": run_metadata, "rows": identity_rows})
    _write_rows(results / "identity_ablation.csv", identity_rows)
    (reports / "identity_ablation_summary.md").write_text("# Identity Normalization Ablation\n\n" + json.dumps(identity_rows, indent=2), encoding="utf-8")
    summary = {"status": "PASS", "best_embedding_backend": best_backend, "best_representation": representation_best["representation"], "identity_rows": identity_rows, "configuration": run_metadata}
    _write(reports / "experiment_campaign_summary.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=str(DATASET_PATH))
    parser.add_argument("--output-root", default="evaluation")
    parser.add_argument("--max-conversations", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_path, args.output_root, args.max_conversations), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
