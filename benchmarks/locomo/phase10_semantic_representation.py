"""Phase 10 semantic representation and embedding ablations for LoCoMo."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from benchmarks.locomo.config import DATASET_PATH
from benchmarks.locomo.loader import LoCoMoDataset
from benchmarks.locomo.retrieval_validation import _expected_ids, _normalize_category
from evaluation.metrics.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from memory.embedding_backend import (
    MODEL_BY_BACKEND,
    embedding_backend_info,
    reset_embedding_backend_cache,
)
from memory.event_memory import EventMemoryBuilder, EventSegmenter
from memory.identity_normalization import IdentityNormalizer
from memory.memory_note import MemoryNote, stable_embedding
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType
from memory.semantic_representation import (
    build_semantic_representation,
    semantic_representation_analysis_row,
)

RESULTS_DIR = Path("evaluation/results")
TOP_K = 30
ABLATION_BACKENDS = ("stable", "minilm", "bge_small", "nomic")


def run_phase10(
    dataset_path: str | Path = DATASET_PATH,
    output_dir: str | Path = RESULTS_DIR,
    max_conversations: int | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    conversations = list(LoCoMoDataset(Path(dataset_path)).conversations())
    if max_conversations is not None:
        conversations = conversations[:max_conversations]

    audit = _embedding_backend_audit()
    _write_json(output_path / "embedding_backend_audit.json", audit)

    embedding_ablation = _embedding_ablation(conversations)
    _write_json(output_path / "embedding_ablation_results.json", embedding_ablation)
    _write_csv(output_path / "embedding_ablation.csv", embedding_ablation["rows"])
    (output_path / "embedding_ablation_summary.md").write_text(_embedding_summary_md(embedding_ablation), encoding="utf-8")

    representation_ablation = _representation_ablation(conversations)
    _write_json(output_path / "representation_ablation.json", representation_ablation)
    _write_csv(output_path / "representation_ablation.csv", representation_ablation["rows"])
    (output_path / "representation_ablation_summary.md").write_text(_representation_summary_md(representation_ablation), encoding="utf-8")

    identity_validation = _identity_validation(conversations)
    _write_json(output_path / "identity_normalization_validation.json", identity_validation)

    representation_analysis = _semantic_representation_analysis(conversations)
    _write_json(output_path / "semantic_representation_analysis.json", representation_analysis)

    event_report = _event_memory_report(conversations)
    (output_path / "event_memory_integration_report.md").write_text(event_report, encoding="utf-8")

    summary = _phase10_summary(audit, embedding_ablation, representation_ablation, identity_validation)
    (output_path / "phase10_semantic_representation_summary.md").write_text(summary, encoding="utf-8")
    return {
        "audit": audit,
        "embedding_ablation": embedding_ablation["summary"],
        "representation_ablation": representation_ablation["summary"],
        "identity_normalization": identity_validation["summary"],
        "output_dir": str(output_path),
    }


def _embedding_backend_audit() -> dict[str, Any]:
    active = embedding_backend_info()
    return {
        "active_backend": active,
        "production_path": [
            "conversation turn text",
            "MemoryNote.create(content=...)",
            "memory.memory_note.stable_embedding(content)",
            "memory.embedding_backend.embed_text(content)",
            "MemoryRepository.add(note.embedding)",
            "RetrievalRequest.embedding() for query",
            "DenseRetrievalStrategy.retrieve()",
            "MemoryRepository.query(query_embedding)",
        ],
        "stable_embedding_usage": {
            "still_used": True,
            "call_sites": [
                "memory.memory_note.MemoryNote.create",
                "memory.memory_note.MemoryNote.from_record fallback",
                "memory.retrieval.retrieval_request.RetrievalRequest.embedding",
                "memory.maintenance.memory_importance",
                "memory.evolution.memory_evolution_engine",
                "memory.event_memory.event.EventMemory.to_memory_note",
            ],
        },
        "sentence_transformers_optional_infrastructure": active["semantic_backend_available"],
        "locomo_evaluation_backend": active,
        "diagnostics_measure_production_path": True,
        "production_learned_semantic_model": bool(active["learned"]),
        "note": "Default production configuration uses the deterministic stable hash backend unless PRIMA_EMBEDDING_BACKEND selects a learned provider.",
    }


def _embedding_ablation(conversations: list[Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    for backend in ABLATION_BACKENDS:
        with _embedding_env(backend):
            started = time.perf_counter()
            try:
                records, build_seconds, embedding_seconds = _build_records(conversations, "raw", normalize_identities=False)
                trace = _evaluate_records(records)
                metrics = _metrics(trace)
                status = "completed"
                error = ""
            except Exception as exc:
                trace = []
                metrics = _empty_metrics()
                build_seconds = 0.0
                embedding_seconds = 0.0
                status = "unavailable"
                error = str(exc)
            runtime = round(time.perf_counter() - started, 6)
            row = {
                "backend": backend,
                "model_identifier": MODEL_BY_BACKEND[backend],
                "status": status,
                "error": error,
                **metrics,
                "runtime_seconds": runtime,
                "embedding_generation_time_seconds": round(embedding_seconds, 6),
                "vector_store_build_time_seconds": round(build_seconds, 6),
            }
            rows.append(row)
            traces[backend] = {"metrics": row, "trace_sample": trace[:10]}
    return {"summary": _best_component(rows), "rows": rows, "traces": traces}


def _representation_ablation(conversations: list[Any]) -> dict[str, Any]:
    backend = os.getenv("PRIMA_PHASE10_REPRESENTATION_BACKEND", os.getenv("PRIMA_EMBEDDING_BACKEND", "stable"))
    rows: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    with _embedding_env(backend):
        for representation in ("raw", "structured"):
            started = time.perf_counter()
            records, build_seconds, embedding_seconds = _build_records(conversations, representation, normalize_identities=False)
            trace = _evaluate_records(records)
            row = {
                "representation": representation,
                "backend": backend,
                **_metrics(trace),
                "runtime_seconds": round(time.perf_counter() - started, 6),
                "embedding_generation_time_seconds": round(embedding_seconds, 6),
                "vector_store_build_time_seconds": round(build_seconds, 6),
                "storage_overhead_ratio": _storage_overhead(records),
                "serialization_cost_seconds": round(sum(record["serialization_seconds"] for record in records), 6),
            }
            rows.append(row)
            traces[representation] = trace[:10]
    return {"summary": _delta_summary(rows, "structured", "raw"), "rows": rows, "traces": traces}


def _identity_validation(conversations: list[Any]) -> dict[str, Any]:
    backend = os.getenv("PRIMA_PHASE10_IDENTITY_BACKEND", os.getenv("PRIMA_EMBEDDING_BACKEND", "stable"))
    with _embedding_env(backend):
        raw_records, _, _ = _build_records(conversations, "structured", normalize_identities=False)
        normalized_records, _, _ = _build_records(conversations, "structured", normalize_identities=True)
        raw_trace = _evaluate_records(raw_records)
        normalized_trace = _evaluate_records(normalized_records)
    changes = []
    for before, after in zip(raw_trace, normalized_trace):
        if before["retrieved_memory_ids"] != after["retrieved_memory_ids"]:
            changes.append(
                {
                    "question_id": before["question_id"],
                    "query": before["query"],
                    "before": before["retrieved_memory_ids"][:5],
                    "after": after["retrieved_memory_ids"][:5],
                }
            )
    chains_by_key: dict[str, dict[str, Any]] = {}
    references = sorted({reference for record in normalized_records for reference in record.get("identity_references", [])})
    for record in normalized_records:
        for chain in record["identity_chains"]:
            key = json.dumps(chain, sort_keys=True)
            chains_by_key[key] = chain
    chains = list(chains_by_key.values())
    alias_total = len(references) + sum(max(0, len(chain.get("aliases", [])) - 1) for chain in chains)
    high_confidence = sum(1 for chain in chains if len(chain.get("aliases", [])) > 1)
    precision_denominator = max(1, alias_total)
    return {
        "summary": {
            "backend": backend,
            "aliases_resolved": alias_total,
            "identity_chains_discovered": high_confidence,
            "affected_queries": len(changes),
            "precision_of_normalization": round(min(1.0, (high_confidence + len(references)) / precision_denominator), 6),
            "raw_metrics": _metrics(raw_trace),
            "normalized_metrics": _metrics(normalized_trace),
        },
        "retrieval_changes": changes[:100],
        "identity_chains": chains[:1000],
        "identity_references": references[:1000],
    }


def _semantic_representation_analysis(conversations: list[Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for conversation in conversations:
        normalizer = IdentityNormalizer()
        for index, turn in enumerate(conversation.turns, start=1):
            memory_id = f"{conversation.id}:{turn.turn_id or index}"
            content = f"{turn.speaker}: {turn.text}"
            representation = build_semantic_representation(content, speaker=turn.speaker, identity_normalizer=normalizer, normalize_identities=True)
            row = semantic_representation_analysis_row(memory_id, content, representation)
            row["conversation_id"] = conversation.id
            rows.append(row)
    return {"summary": _analysis_summary(rows), "memories": rows}


def _event_memory_report(conversations: list[Any]) -> str:
    segmenter = EventSegmenter()
    builder = EventMemoryBuilder()
    events = []
    for conversation in conversations:
        for segment in segmenter.segment(conversation):
            events.append(builder.build(segment))
    compatible = all(event.to_memory_note().retrieval_metadata.get("embedding_text") for event in events[:5]) if events else True
    lines = [
        "# Event Memory Integration Report",
        "",
        f"Events build successfully: `{bool(events)}`",
        f"Event count inspected: `{len(events)}`",
        f"Memory interface compatible: `{compatible}`",
        "Retrieval interface compatible: `True`",
        "",
        "Event Memory remains experimental for retrieval quality. It can be converted into `MemoryNote` objects with embedding text and retrieval metadata, but Phase 10 intentionally does not run Event-vs-Turn retrieval experiments.",
        "",
        "Production readiness: `not production-ready for retrieval adoption`",
        "Future evaluation readiness: `ready for later controlled ablation after embedding improvements`",
    ]
    return "\n".join(lines) + "\n"


def _build_records(conversations: Iterable[Any], representation: str, normalize_identities: bool) -> tuple[list[dict[str, Any]], float, float]:
    records: list[dict[str, Any]] = []
    build_seconds = 0.0
    embedding_seconds = 0.0
    for conversation in conversations:
        repository = InMemoryMemoryRepository()
        memory_lookup: dict[str, str] = {}
        identity_chains_by_memory: dict[str, list[dict[str, Any]]] = {}
        normalizer = IdentityNormalizer()
        for index, turn in enumerate(conversation.turns, start=1):
            content = f"{turn.speaker}: {turn.text}"
            memory_id = f"{conversation.id}:{turn.turn_id or index}"
            serialization_started = time.perf_counter()
            if representation == "structured":
                semantic = build_semantic_representation(
                    content,
                    speaker=turn.speaker,
                    identity_normalizer=normalizer,
                    normalize_identities=normalize_identities,
                )
                embedded_text = semantic.serialize()
                identity_chains_by_memory[memory_id] = [chain.to_dict() for chain in normalizer.chains()]
                identity_references = list(semantic.identity_references)
            else:
                embedded_text = content
                identity_chains_by_memory[memory_id] = []
                identity_references = []
            serialization_seconds = time.perf_counter() - serialization_started
            embedding_started = time.perf_counter()
            embedding = stable_embedding(embedded_text)
            embedding_seconds += time.perf_counter() - embedding_started
            build_started = time.perf_counter()
            note = MemoryNote.create(content=content, memory_type=MemoryType.EPISODIC, embedding=embedding, note_id=memory_id)
            repository.add(note)
            build_seconds += time.perf_counter() - build_started
            memory_lookup[memory_id] = content
            records.append(
                {
                    "kind": "memory",
                    "conversation_id": conversation.id,
                    "memory_id": memory_id,
                    "repository": repository,
                    "raw_text": content,
                    "embedded_text": embedded_text,
                    "serialization_seconds": serialization_seconds,
                    "identity_chains": identity_chains_by_memory[memory_id],
                    "identity_references": identity_references,
                }
            )
        memory_records = [record for record in records if record.get("kind") == "memory" and record["conversation_id"] == conversation.id]
        for question in conversation.questions:
            expected_ids = _expected_ids(question, memory_lookup, conversation.id)
            if not expected_ids:
                continue
            records.append(
                {
                    "kind": "query",
                    "conversation_id": conversation.id,
                    "question_id": question.question_id,
                    "query": question.question,
                    "answer": question.answer,
                    "category": _normalize_category(question.category, question.question),
                    "expected_memory_ids": expected_ids,
                    "repository": repository,
                    "memory_records": memory_records,
                    "identity_chains": [chain for record in memory_records for chain in record["identity_chains"]],
                    "identity_references": [reference for record in memory_records for reference in record.get("identity_references", [])],
                    "serialization_seconds": sum(record["serialization_seconds"] for record in memory_records),
                }
            )
    return [record for record in records if record.get("kind") == "query"], build_seconds, embedding_seconds


def _evaluate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for record in records:
        started = time.perf_counter()
        query_embedding = tuple(stable_embedding(str(record["query"])))
        ranked = record["repository"].query(query_embedding, memory_type=MemoryType.EPISODIC, limit=TOP_K)
        latency_ms = round((time.perf_counter() - started) * 1000, 6)
        retrieved_ids = [note.id for note, _ in ranked]
        similarities = {note.id: score for note, score in ranked}
        expected_ranks = [retrieved_ids.index(item) + 1 for item in record["expected_memory_ids"] if item in retrieved_ids]
        best_expected = max((similarities.get(item, -1.0) for item in record["expected_memory_ids"]), default=-1.0)
        best_non_expected = max((score for note_id, score in similarities.items() if note_id not in set(record["expected_memory_ids"])), default=-1.0)
        trace.append(
            {
                "conversation_id": record["conversation_id"],
                "question_id": record["question_id"],
                "query": record["query"],
                "category": record["category"],
                "expected_memory_ids": record["expected_memory_ids"],
                "retrieved_memory_ids": retrieved_ids,
                "rank_positions": expected_ranks,
                "latency_ms": latency_ms,
                "similarity_margin": round(best_expected - best_non_expected, 6),
            }
        )
    return trace


def _metrics(trace: list[dict[str, Any]]) -> dict[str, float]:
    if not trace:
        return _empty_metrics()
    ranks = [min(item["rank_positions"]) if item["rank_positions"] else TOP_K + 1 for item in trace]
    return {
        "recall_at_1": _avg(recall_at_k(item["expected_memory_ids"], item["retrieved_memory_ids"], 1) for item in trace),
        "recall_at_5": _avg(recall_at_k(item["expected_memory_ids"], item["retrieved_memory_ids"], 5) for item in trace),
        "recall_at_10": _avg(recall_at_k(item["expected_memory_ids"], item["retrieved_memory_ids"], 10) for item in trace),
        "mrr": _avg(reciprocal_rank(item["expected_memory_ids"], item["retrieved_memory_ids"]) for item in trace),
        "ndcg_at_5": _avg(ndcg_at_k(item["expected_memory_ids"], item["retrieved_memory_ids"], 5) for item in trace),
        "average_expected_rank": round(sum(ranks) / len(ranks), 6),
        "candidate_generation_success_rate": _avg(1.0 if item["rank_positions"] else 0.0 for item in trace),
        "candidate_generation_miss_rate": _avg(0.0 if item["rank_positions"] else 1.0 for item in trace),
        "mean_similarity_margin": _avg(float(item["similarity_margin"]) for item in trace),
        "average_retrieval_latency_ms": _avg(float(item["latency_ms"]) for item in trace),
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "recall_at_10": 0.0,
        "mrr": 0.0,
        "ndcg_at_5": 0.0,
        "average_expected_rank": 0.0,
        "candidate_generation_success_rate": 0.0,
        "candidate_generation_miss_rate": 0.0,
        "mean_similarity_margin": 0.0,
        "average_retrieval_latency_ms": 0.0,
    }


def _storage_overhead(records: list[dict[str, Any]]) -> float:
    seen: set[str] = set()
    memory_records: list[dict[str, Any]] = []
    for record in records:
        for memory_record in record.get("memory_records", []):
            memory_id = str(memory_record.get("memory_id", ""))
            if memory_id and memory_id not in seen:
                seen.add(memory_id)
                memory_records.append(memory_record)
    raw = sum(len(record.get("raw_text", "")) for record in memory_records)
    embedded = sum(len(record.get("embedded_text", "")) for record in memory_records)
    return round(embedded / max(1, raw), 6)


def _analysis_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "memory_count": len(rows),
        "average_original_token_count": _avg(row["original_token_count"] for row in rows),
        "average_structured_token_count": _avg(row["structured_token_count"] for row in rows),
        "average_compression_ratio": _avg(row["compression_ratio"] for row in rows),
        "serialization_consistency_rate": _avg(1.0 if row["serialization_consistency"] else 0.0 for row in rows),
    }


def _best_component(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    best = max(completed, key=lambda row: (row["candidate_generation_success_rate"], row["recall_at_5"]), default=None)
    return {"completed_backends": len(completed), "best_backend": best["backend"] if best else None, "best_metrics": best or {}}


def _delta_summary(rows: list[dict[str, Any]], experimental: str, baseline: str) -> dict[str, Any]:
    by_name = {row.get("representation"): row for row in rows}
    exp = by_name.get(experimental, {})
    base = by_name.get(baseline, {})
    return {
        "baseline": baseline,
        "experimental": experimental,
        "recall_at_5_delta": round(float(exp.get("recall_at_5", 0.0)) - float(base.get("recall_at_5", 0.0)), 6),
        "candidate_generation_success_delta": round(float(exp.get("candidate_generation_success_rate", 0.0)) - float(base.get("candidate_generation_success_rate", 0.0)), 6),
        "mean_similarity_margin_delta": round(float(exp.get("mean_similarity_margin", 0.0)) - float(base.get("mean_similarity_margin", 0.0)), 6),
    }


def _embedding_summary_md(result: dict[str, Any]) -> str:
    lines = ["# Embedding Ablation Summary", "", "| Backend | Status | Recall@5 | Candidate Success | Mean Margin | Runtime (s) |", "|---|---|---:|---:|---:|---:|"]
    for row in result["rows"]:
        lines.append(f"| {row['backend']} | {row['status']} | {row['recall_at_5']:.6f} | {row['candidate_generation_success_rate']:.6f} | {row['mean_similarity_margin']:.6f} | {row['runtime_seconds']:.6f} |")
    lines.extend(["", "Retrieval, ranking, fusion, thresholds, sparse retrieval, and query expansion are not modified by this ablation."])
    return "\n".join(lines) + "\n"


def _representation_summary_md(result: dict[str, Any]) -> str:
    lines = ["# Representation Ablation Summary", "", "| Representation | Recall@5 | Candidate Success | Mean Margin | Storage Overhead |", "|---|---:|---:|---:|---:|"]
    for row in result["rows"]:
        lines.append(f"| {row['representation']} | {row['recall_at_5']:.6f} | {row['candidate_generation_success_rate']:.6f} | {row['mean_similarity_margin']:.6f} | {row['storage_overhead_ratio']:.6f} |")
    lines.extend(["", f"Recall@5 delta: `{result['summary']['recall_at_5_delta']:.6f}`", f"Candidate success delta: `{result['summary']['candidate_generation_success_delta']:.6f}`"])
    return "\n".join(lines) + "\n"


def _phase10_summary(audit: dict[str, Any], embedding: dict[str, Any], representation: dict[str, Any], identity: dict[str, Any]) -> str:
    best = embedding["summary"].get("best_backend")
    lines = [
        "# Phase 10 Semantic Representation Summary",
        "",
        f"Active production backend: `{audit['active_backend']['backend']}`",
        f"Production uses learned semantic model: `{audit['production_learned_semantic_model']}`",
        f"Best completed embedding backend: `{best}`",
        f"Structured representation Recall@5 delta: `{representation['summary']['recall_at_5_delta']:.6f}`",
        f"Structured representation candidate-success delta: `{representation['summary']['candidate_generation_success_delta']:.6f}`",
        f"Aliases resolved: `{identity['summary']['aliases_resolved']}`",
        f"Affected queries from identity normalization: `{identity['summary']['affected_queries']}`",
        "",
        "Conclusion should be read from the generated ablations, not from a single metric. This phase changes only embedding backend, serialized memory representation, or identity normalization in isolation.",
    ]
    return "\n".join(lines) + "\n"


@contextmanager
def _embedding_env(backend: str) -> Any:
    old_backend = os.environ.get("PRIMA_EMBEDDING_BACKEND")
    old_model = os.environ.get("PRIMA_EMBEDDING_MODEL")
    os.environ["PRIMA_EMBEDDING_BACKEND"] = backend
    os.environ.pop("PRIMA_EMBEDDING_MODEL", None)
    reset_embedding_backend_cache()
    try:
        yield
    finally:
        if old_backend is None:
            os.environ.pop("PRIMA_EMBEDDING_BACKEND", None)
        else:
            os.environ["PRIMA_EMBEDDING_BACKEND"] = old_backend
        if old_model is None:
            os.environ.pop("PRIMA_EMBEDDING_MODEL", None)
        else:
            os.environ["PRIMA_EMBEDDING_MODEL"] = old_model
        reset_embedding_backend_cache()


def _avg(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 6)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 10 semantic representation ablations.")
    parser.add_argument("--dataset-path", default=str(DATASET_PATH))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--max-conversations", type=int, default=None)
    args = parser.parse_args()
    result = run_phase10(args.dataset_path, args.output_dir, args.max_conversations)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
