"""Semantic retrieval validation runner for Phase 4.1."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from evaluation.metrics.retrieval_metrics import summarize_retrieval_metrics
from memory.graph.graph_edge import GraphEdge
from memory.graph.graph_node import GraphNode
from memory.graph.graph_repository import GraphRepository
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryLevel, MemoryType
from memory.retrieval.dense_strategy import DenseRetrievalStrategy
from memory.retrieval.graph_strategy import GraphTraversalStrategy
from memory.retrieval.retrieval_controller import RetrievalController
from memory.retrieval.retrieval_request import RetrievalRequest
from memory.retrieval.retrieval_strategy import RetrievalStrategy
from memory.retrieval.sparse_strategy import SparseRetrievalStrategy
from memory.retrieval.temporal_strategy import TemporalRetrievalStrategy


DEFAULT_GOLD_PATH = Path("evaluation/datasets/retrieval_gold_v2.json")
DEFAULT_REPORT_PATH = Path("evaluation/results/retrieval_semantic_results.json")
DEFAULT_METRICS_PATH = Path("evaluation/results/retrieval_semantic_metrics.json")
DEFAULT_CATEGORY_PATH = Path("evaluation/results/retrieval_category_metrics.json")
DEFAULT_ABLATION_PATH = Path("evaluation/results/retrieval_ablation_v2.json")
DEFAULT_CONTRIBUTION_PATH = Path("evaluation/results/retrieval_component_contribution.json")
DEFAULT_FAILURES_PATH = Path("evaluation/results/retrieval_failures_v2.json")
DEFAULT_HARD_QUERIES_PATH = Path("evaluation/results/hard_queries.json")
DEFAULT_TRACE_PATH = Path("evaluation/results/retrieval_semantic_trace.json")


@dataclass(frozen=True, slots=True)
class SemanticRetrievalConfiguration:
    """Named retrieval strategy configuration."""

    name: str
    strategies: tuple[RetrievalStrategy, ...]


class SemanticRetrievalRunner:
    """Evaluate realistic semantic retrieval quality without changing retrieval code."""

    def __init__(
        self,
        gold_path: str | Path = DEFAULT_GOLD_PATH,
        report_path: str | Path = DEFAULT_REPORT_PATH,
        metrics_path: str | Path = DEFAULT_METRICS_PATH,
        category_path: str | Path = DEFAULT_CATEGORY_PATH,
        ablation_path: str | Path = DEFAULT_ABLATION_PATH,
        contribution_path: str | Path = DEFAULT_CONTRIBUTION_PATH,
        failures_path: str | Path = DEFAULT_FAILURES_PATH,
        hard_queries_path: str | Path = DEFAULT_HARD_QUERIES_PATH,
        trace_path: str | Path = DEFAULT_TRACE_PATH,
        top_k: int = 5,
    ) -> None:
        self.gold_path = Path(gold_path)
        self.report_path = Path(report_path)
        self.metrics_path = Path(metrics_path)
        self.category_path = Path(category_path)
        self.ablation_path = Path(ablation_path)
        self.contribution_path = Path(contribution_path)
        self.failures_path = Path(failures_path)
        self.hard_queries_path = Path(hard_queries_path)
        self.trace_path = Path(trace_path)
        self.top_k = top_k

    def run(self) -> dict[str, Any]:
        """Run semantic benchmark, ablations, category reports, and diagnostics."""
        gold_records = self._load_gold()
        repository = self._build_repository(gold_records)
        graph_repository = self._build_graph(gold_records)
        configurations = self._configurations(graph_repository)

        traces_by_config: dict[str, list[dict[str, Any]]] = {}
        ablation: dict[str, dict[str, float]] = {}
        for config in configurations:
            traces = self._run_configuration(config, repository, gold_records)
            traces_by_config[config.name] = traces
            ablation[config.name] = summarize_retrieval_metrics(traces)

        full_name = "dense_sparse_temporal_graph"
        full_trace = traces_by_config[full_name]
        metrics = summarize_retrieval_metrics(full_trace)
        category_metrics = self._category_metrics(full_trace)
        contribution = self._component_contribution(ablation, traces_by_config)
        failures = self._failure_analysis(full_trace)
        hard_queries = self._hard_queries(full_trace)

        report = {
            "dataset_size": len(gold_records),
            "metrics": metrics,
            "category_metrics": category_metrics,
            "ablation": ablation,
            "component_contribution": contribution,
            "failure_count": len(failures),
            "hard_query_count": len(hard_queries),
        }
        self._write_json(self.report_path, report)
        self._write_json(self.metrics_path, metrics)
        self._write_json(self.category_path, category_metrics)
        self._write_json(self.ablation_path, ablation)
        self._write_json(self.contribution_path, contribution)
        self._write_json(self.failures_path, failures)
        self._write_json(self.hard_queries_path, hard_queries)
        self._write_json(self.trace_path, full_trace)
        return report

    def _run_configuration(
        self,
        config: SemanticRetrievalConfiguration,
        repository: InMemoryMemoryRepository,
        gold_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        controller = RetrievalController(repository=repository, strategies=list(config.strategies))
        traces: list[dict[str, Any]] = []
        for record in gold_records:
            start = time.perf_counter()
            response = controller.retrieve(self._request(record))
            latency_ms = round((time.perf_counter() - start) * 1000, 6)
            retrieved_ids = [result.note.id for result in response.results]
            expected_ids = [str(item) for item in record["expected_memory_ids"]]
            rank_positions = [
                retrieved_ids.index(expected_id) + 1
                for expected_id in expected_ids
                if expected_id in retrieved_ids
            ]
            traces.append(
                {
                    "query": str(record["query"]),
                    "expected_memory_ids": expected_ids,
                    "retrieved_memory_ids": retrieved_ids,
                    "rank_positions": rank_positions,
                    "hit": bool(rank_positions),
                    "latency_ms": latency_ms,
                    "retrieval_strategy": config.name,
                    "category": str(record.get("category", "")),
                    "semantic_relation": str(record.get("semantic_relation", "")),
                    "strategy_scores": [
                        {
                            "memory_id": result.note.id,
                            "score": result.score,
                            "strategy_scores": dict(result.strategy_scores),
                        }
                        for result in response.results
                    ],
                }
            )
        return traces

    def _request(self, record: dict[str, Any]) -> RetrievalRequest:
        memory_type = MemoryType(str(record.get("memory_type", MemoryType.EPISODIC.value)))
        return RetrievalRequest(
            query=str(record["query"]),
            memory_types=(memory_type,),
            top_k=self.top_k,
        )

    def _load_gold(self) -> list[dict[str, Any]]:
        records = json.loads(self.gold_path.read_text(encoding="utf-8-sig"))
        if not 150 <= len(records) <= 250:
            raise ValueError("retrieval_gold_v2.json must contain 150-250 records.")
        required_categories = {"preference", "identity", "temporal", "emotional", "multihop", "graph"}
        categories = {str(record.get("category", "")) for record in records}
        missing = required_categories - categories
        if missing:
            raise ValueError(f"retrieval_gold_v2.json is missing categories: {sorted(missing)}")
        for record in records:
            if "query" not in record or "memory_text" not in record or "expected_memory_ids" not in record:
                raise ValueError("Each semantic gold record must include query, memory_text, and expected_memory_ids.")
        return records

    def _build_repository(self, records: list[dict[str, Any]]) -> InMemoryMemoryRepository:
        repository = InMemoryMemoryRepository()
        now = datetime.now(timezone.utc)
        for record in records:
            memory_type = MemoryType(str(record.get("memory_type", MemoryType.EPISODIC.value)))
            memory_level = (
                MemoryLevel.SEMANTIC_ABSTRACTION
                if memory_type == MemoryType.SEMANTIC
                else MemoryLevel.EMOTIONAL_TRACE
                if memory_type == MemoryType.EMOTIONAL
                else MemoryLevel.EPISODIC_EVENT
            )
            timestamp = now - timedelta(hours=float(record.get("timestamp_offset_hours", 240.0)))
            note = MemoryNote.create(
                content=str(record["memory_text"]),
                memory_type=memory_type,
                memory_level=memory_level,
                context={
                    "category": record.get("category", ""),
                    "semantic_relation": record.get("semantic_relation", ""),
                    "source": "retrieval_gold_v2",
                },
                note_id=str(record.get("memory_id") or record["expected_memory_ids"][0]),
                salience_score=float(record.get("salience_score", 0.8)),
                retention_score=1.0,
            ).with_updates(timestamp=timestamp)
            repository.add(note)
        return repository

    def _build_graph(self, records: list[dict[str, Any]]) -> GraphRepository:
        graph_repository = GraphRepository()
        ids = {str(record.get("memory_id") or record["expected_memory_ids"][0]) for record in records}
        for memory_id in ids:
            graph_repository.add_node(GraphNode(id=f"node_{memory_id}", memory_id=memory_id))
        for record in records:
            source_id = str(record.get("memory_id") or record["expected_memory_ids"][0])
            for target_id in record.get("graph_links", []):
                target = str(target_id)
                if target not in ids:
                    continue
                graph_repository.add_edge(
                    GraphEdge(
                        source_id=f"node_{source_id}",
                        target_id=f"node_{target}",
                        relation_type="semantic_gold_link",
                        weight=1.0,
                    )
                )
        return graph_repository

    def _configurations(self, graph_repository: GraphRepository) -> tuple[SemanticRetrievalConfiguration, ...]:
        dense = DenseRetrievalStrategy()
        sparse = SparseRetrievalStrategy()
        temporal = TemporalRetrievalStrategy()
        graph = GraphTraversalStrategy(graph_repository=graph_repository)
        return (
            SemanticRetrievalConfiguration("dense_only", (dense,)),
            SemanticRetrievalConfiguration("dense_sparse", (dense, sparse)),
            SemanticRetrievalConfiguration("dense_sparse_temporal", (dense, sparse, temporal)),
            SemanticRetrievalConfiguration("dense_sparse_temporal_graph", (dense, sparse, temporal, graph)),
        )

    def _category_metrics(self, traces: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        categories = sorted({str(trace.get("category", "")) for trace in traces})
        return {
            category: summarize_retrieval_metrics(
                [trace for trace in traces if str(trace.get("category", "")) == category]
            )
            for category in categories
        }

    def _component_contribution(
        self,
        ablation: dict[str, dict[str, float]],
        traces_by_config: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        dense = ablation["dense_only"]
        dense_sparse = ablation["dense_sparse"]
        dense_sparse_temporal = ablation["dense_sparse_temporal"]
        full = ablation["dense_sparse_temporal_graph"]
        return {
            "overall": {
                "sparse_gain": round(dense_sparse["recall_at_5"] - dense["recall_at_5"], 6),
                "temporal_gain": round(dense_sparse_temporal["recall_at_5"] - dense_sparse["recall_at_5"], 6),
                "graph_gain": round(full["recall_at_5"] - dense_sparse_temporal["recall_at_5"], 6),
                "full_stack_gain": round(full["recall_at_5"] - dense["recall_at_5"], 6),
            },
            "by_category": {
                category: self._category_contribution(category, traces_by_config)
                for category in sorted({trace["category"] for trace in traces_by_config["dense_only"]})
            },
        }

    def _category_contribution(
        self,
        category: str,
        traces_by_config: dict[str, list[dict[str, Any]]],
    ) -> dict[str, float]:
        category_metrics = {
            name: summarize_retrieval_metrics([trace for trace in traces if trace["category"] == category])
            for name, traces in traces_by_config.items()
        }
        return {
            "sparse_gain": round(
                category_metrics["dense_sparse"]["recall_at_5"]
                - category_metrics["dense_only"]["recall_at_5"],
                6,
            ),
            "temporal_gain": round(
                category_metrics["dense_sparse_temporal"]["recall_at_5"]
                - category_metrics["dense_sparse"]["recall_at_5"],
                6,
            ),
            "graph_gain": round(
                category_metrics["dense_sparse_temporal_graph"]["recall_at_5"]
                - category_metrics["dense_sparse_temporal"]["recall_at_5"],
                6,
            ),
        }

    def _failure_analysis(self, traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        for trace in traces:
            if trace["hit"]:
                continue
            failures.append(
                {
                    "query": trace["query"],
                    "expected_memory_ids": trace["expected_memory_ids"],
                    "retrieved_memory_ids": trace["retrieved_memory_ids"],
                    "failure_type": self._failure_type(trace),
                }
            )
        return failures

    def _failure_type(self, trace: dict[str, Any]) -> str:
        if not trace["retrieved_memory_ids"]:
            return "NO_MEMORY_FOUND"
        category = str(trace.get("category", ""))
        if category == "graph":
            return "GRAPH_MISS"
        if category == "temporal":
            return "TEMPORAL_MISS"
        if category == "multihop":
            return "MULTIHOP_MISS"
        if trace["rank_positions"]:
            return "LOW_RANK_MEMORY"
        return "WRONG_MEMORY"

    def _hard_queries(self, traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hard: list[dict[str, Any]] = []
        for trace in traces:
            if not trace["rank_positions"]:
                continue
            best_rank = min(trace["rank_positions"])
            if 1 < best_rank <= self.top_k:
                hard.append(
                    {
                        "query": trace["query"],
                        "expected_memory_ids": trace["expected_memory_ids"],
                        "retrieved_memory_ids": trace["retrieved_memory_ids"],
                        "best_rank": best_rank,
                        "category": trace["category"],
                    }
                )
        return hard

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_semantic_retrieval_benchmark() -> dict[str, Any]:
    """Run the default Phase 4.1 semantic retrieval benchmark."""
    return SemanticRetrievalRunner().run()


if __name__ == "__main__":
    run_semantic_retrieval_benchmark()
