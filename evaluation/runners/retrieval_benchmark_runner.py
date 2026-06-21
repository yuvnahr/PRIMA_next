"""Retrieval benchmark and ablation runner."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.metrics.retrieval_metrics import summarize_retrieval_metrics
from memory.graph.graph_builder import GraphBuilder
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


DEFAULT_GOLD_PATH = Path("evaluation/datasets/retrieval_gold.json")
DEFAULT_RESULTS_PATH = Path("evaluation/results/retrieval_benchmark_results.json")
DEFAULT_TRACE_PATH = Path("evaluation/results/retrieval_trace.json")
DEFAULT_ABLATION_PATH = Path("evaluation/results/retrieval_ablation.json")
DEFAULT_FAILURES_PATH = Path("evaluation/results/retrieval_failures.json")


@dataclass(frozen=True, slots=True)
class RetrievalConfiguration:
    """Named retrieval strategy configuration."""

    name: str
    strategies: tuple[RetrievalStrategy, ...]


class RetrievalBenchmarkRunner:
    """Run retrieval quality validation against a gold dataset."""

    def __init__(
        self,
        gold_path: str | Path = DEFAULT_GOLD_PATH,
        results_path: str | Path = DEFAULT_RESULTS_PATH,
        trace_path: str | Path = DEFAULT_TRACE_PATH,
        ablation_path: str | Path = DEFAULT_ABLATION_PATH,
        failures_path: str | Path = DEFAULT_FAILURES_PATH,
        top_k: int = 5,
    ) -> None:
        self.gold_path = Path(gold_path)
        self.results_path = Path(results_path)
        self.trace_path = Path(trace_path)
        self.ablation_path = Path(ablation_path)
        self.failures_path = Path(failures_path)
        self.top_k = top_k

    def run(self) -> dict[str, Any]:
        """Execute benchmark, ablations, contribution analysis, and failure analysis."""
        gold_records = self._load_gold()
        repository = self._build_repository(gold_records)
        graph_repository = self._build_graph(repository)
        configurations = self._configurations(graph_repository)
        ablation: dict[str, dict[str, float]] = {}
        traces_by_config: dict[str, list[dict[str, Any]]] = {}

        for config in configurations:
            traces = self._run_configuration(config, repository, gold_records)
            traces_by_config[config.name] = traces
            ablation[config.name] = summarize_retrieval_metrics(traces)

        full_name = "dense_sparse_temporal_graph"
        full_trace = traces_by_config[full_name]
        benchmark = {
            "dataset_size": len(gold_records),
            "top_k": self.top_k,
            "metrics": summarize_retrieval_metrics(full_trace),
            "ablation": ablation,
            "contribution_analysis": self._contribution_analysis(ablation),
        }
        self._write_json(self.results_path, benchmark)
        self._write_json(self.trace_path, full_trace)
        self._write_json(self.ablation_path, ablation)
        self._write_json(self.failures_path, self._failure_analysis(full_trace, full_name))
        return benchmark

    def _run_configuration(
        self,
        config: RetrievalConfiguration,
        repository: InMemoryMemoryRepository,
        gold_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        controller = RetrievalController(repository=repository, strategies=list(config.strategies))
        traces: list[dict[str, Any]] = []
        for record in gold_records:
            start = time.perf_counter()
            response = controller.retrieve(RetrievalRequest(query=str(record["query"]), top_k=self.top_k))
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

    def _load_gold(self) -> list[dict[str, Any]]:
        records = json.loads(self.gold_path.read_text(encoding="utf-8-sig"))
        if len(records) < 100:
            raise ValueError("retrieval_gold.json must contain at least 100 records.")
        for record in records:
            if "query" not in record or "expected_memory_ids" not in record:
                raise ValueError("Each gold record must include query and expected_memory_ids.")
        return records

    def _build_repository(self, records: list[dict[str, Any]]) -> InMemoryMemoryRepository:
        repository = InMemoryMemoryRepository()
        for record in records:
            memory_type = MemoryType(str(record.get("memory_type", MemoryType.EPISODIC.value)))
            memory_level = (
                MemoryLevel.SEMANTIC_ABSTRACTION
                if memory_type == MemoryType.SEMANTIC
                else MemoryLevel.EPISODIC_EVENT
            )
            repository.add(
                MemoryNote.create(
                    content=str(record.get("memory_text") or record["query"]),
                    memory_type=memory_type,
                    memory_level=memory_level,
                    context={"category": record.get("category", ""), "source": "retrieval_gold"},
                    note_id=str(record.get("memory_id") or record["expected_memory_ids"][0]),
                    salience_score=0.8,
                    retention_score=1.0,
                )
            )
        return repository

    def _build_graph(self, repository: InMemoryMemoryRepository) -> GraphRepository:
        graph_repository = GraphRepository()
        builder = GraphBuilder(graph_repository)
        for memory_type in MemoryType:
            builder.rebuild_for_type(repository, memory_type)
        return graph_repository

    def _configurations(self, graph_repository: GraphRepository) -> tuple[RetrievalConfiguration, ...]:
        dense = DenseRetrievalStrategy()
        sparse = SparseRetrievalStrategy()
        temporal = TemporalRetrievalStrategy()
        graph = GraphTraversalStrategy(graph_repository=graph_repository)
        return (
            RetrievalConfiguration("dense_only", (dense,)),
            RetrievalConfiguration("dense_sparse", (dense, sparse)),
            RetrievalConfiguration("dense_sparse_temporal", (dense, sparse, temporal)),
            RetrievalConfiguration("dense_sparse_temporal_graph", (dense, sparse, temporal, graph)),
        )

    def _contribution_analysis(self, ablation: dict[str, dict[str, float]]) -> dict[str, float]:
        dense = ablation["dense_only"]
        dense_sparse = ablation["dense_sparse"]
        dense_sparse_temporal = ablation["dense_sparse_temporal"]
        full = ablation["dense_sparse_temporal_graph"]
        return {
            "sparse_recall_at_5_gain": round(dense_sparse["recall_at_5"] - dense["recall_at_5"], 6),
            "temporal_recall_at_5_gain": round(
                dense_sparse_temporal["recall_at_5"] - dense_sparse["recall_at_5"],
                6,
            ),
            "graph_recall_at_5_gain": round(
                full["recall_at_5"] - dense_sparse_temporal["recall_at_5"],
                6,
            ),
            "full_stack_recall_at_5_gain": round(full["recall_at_5"] - dense["recall_at_5"], 6),
            "full_stack_mrr_gain": round(full["mrr"] - dense["mrr"], 6),
        }

    def _failure_analysis(self, traces: list[dict[str, Any]], strategy_name: str) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        for trace in traces:
            if trace["hit"] and min(trace["rank_positions"]) <= self.top_k:
                continue
            category = "NO_MEMORY_FOUND"
            if trace["retrieved_memory_ids"]:
                category = "LOW_RANK_RELEVANT_MEMORY" if trace["rank_positions"] else "WRONG_MEMORY"
            if "graph" in strategy_name and not trace["hit"]:
                category = "GRAPH_MISS"
            elif "temporal" in strategy_name and not trace["hit"]:
                category = "TEMPORAL_MISS"
            failures.append(
                {
                    "query": trace["query"],
                    "expected_memory": trace["expected_memory_ids"],
                    "retrieved_memories": trace["retrieved_memory_ids"],
                    "retrieval_strategy_used": strategy_name,
                    "failure_category": category,
                }
            )
        return failures

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_retrieval_benchmark() -> dict[str, Any]:
    """Run the default Phase 4 retrieval benchmark."""
    return RetrievalBenchmarkRunner().run()


if __name__ == "__main__":
    run_retrieval_benchmark()
