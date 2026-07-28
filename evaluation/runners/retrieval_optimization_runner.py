"""Phase 4.2 retrieval optimization and diagnostics runner."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.metrics.retrieval_metrics import (
    summarize_retrieval_metrics,
)
from evaluation.runners.retrieval_semantic_runner import SemanticRetrievalConfiguration, SemanticRetrievalRunner
from memory.memory_note import stable_embedding
from memory.retrieval.hybrid_fusion import HybridFusion, HybridFusionConfig
from memory.retrieval.reranker import Reranker
from memory.retrieval.retrieval_controller import RetrievalController
from memory.retrieval.retrieval_result import RetrievalResult
from memory.retrieval.retrieval_router import RetrievalRouter


class PassthroughReranker(Reranker):
    """Reranker used to measure hybrid ranking before reranking."""

    def rerank(self, results: list[RetrievalResult], request: Any | None = None) -> list[RetrievalResult]:
        return results


class RetrievalOptimizationRunner:
    """Generate Phase 4.2 optimization and diagnostics artifacts."""

    def __init__(
        self,
        gold_path: str | Path = "evaluation/datasets/retrieval_gold_v2.json",
        results_dir: str | Path = "evaluation/results",
    ) -> None:
        self.semantic_runner = SemanticRetrievalRunner(gold_path=gold_path)
        self.results_dir = Path(results_dir)

    def run(self) -> dict[str, Any]:
        """Run all Phase 4.2 analyses."""
        records = self.semantic_runner._load_gold()
        repository = self.semantic_runner._build_repository(records)
        graph_repository = self.semantic_runner._build_graph(records)

        baseline_trace = self._trace_with_weights(records, repository, graph_repository, {
            "dense": 0.4,
            "sparse": 0.3,
            "temporal": 0.15,
            "graph": 0.15,
        })
        optimized_trace = self._trace_with_weights(records, repository, graph_repository, {
            "dense": 0.20,
            "sparse": 0.55,
            "temporal": 0.15,
            "graph": 0.10,
        })
        routed_trace, routing_log = self._routed_trace(records, repository, graph_repository)
        best_search = self._fusion_weight_search(records, repository, graph_repository)
        best_trace = best_search["best_trace"]

        failure_analysis = self._failure_analysis(best_trace)
        embedding_diagnostics = self._embedding_diagnostics(records, repository, best_trace)
        graph_diagnostics = self._graph_diagnostics(records, graph_repository, best_trace)
        temporal_diagnostics = self._temporal_diagnostics(records, best_trace)
        reranker_analysis = self._reranker_analysis(records, repository, graph_repository)
        hard_query_analysis = self._hard_query_analysis(best_trace)
        optimization_results = self._optimization_results(baseline_trace, optimized_trace, routed_trace, best_trace)
        component_contribution = self._component_contribution(records, repository, graph_repository, best_search["best_weights"])

        self._write("retrieval_failure_analysis.json", failure_analysis)
        self._write("embedding_diagnostics.json", embedding_diagnostics)
        self._write("fusion_weight_search.json", {k: v for k, v in best_search.items() if k != "best_trace"})
        self._write("graph_diagnostics.json", graph_diagnostics)
        self._write("temporal_diagnostics.json", temporal_diagnostics)
        self._write("reranker_analysis.json", reranker_analysis)
        self._write("retrieval_routing.json", {"routing_decisions": routing_log, "metrics": summarize_retrieval_metrics(routed_trace)})
        self._write("hard_query_analysis.json", hard_query_analysis)
        self._write("retrieval_optimization_results.json", optimization_results)
        self._write("component_contribution_v2.json", component_contribution)
        return optimization_results

    def _trace_with_weights(
        self,
        records: list[dict[str, Any]],
        repository: Any,
        graph_repository: Any,
        weights: dict[str, float],
        reranker: Reranker | None = None,
    ) -> list[dict[str, Any]]:
        config = SemanticRetrievalConfiguration(
            "weighted_full_stack",
            self.semantic_runner._configurations(graph_repository)[-1].strategies,
        )
        controller = RetrievalController(
            repository=repository,
            strategies=list(config.strategies),
            fusion=HybridFusion(HybridFusionConfig(weights=weights)),
            reranker=reranker,
        )
        traces: list[dict[str, Any]] = []
        for record in records:
            response = controller.retrieve(self.semantic_runner._request(record))
            traces.append(self._trace_record(record, response.results, "weighted_full_stack", 0.0))
        return traces

    def _routed_trace(self, records: list[dict[str, Any]], repository: Any, graph_repository: Any) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
        router = RetrievalRouter(graph_repository)
        traces: list[dict[str, Any]] = []
        routing_log: list[dict[str, object]] = []
        for record in records:
            decision = router.route(str(record["query"]))
            controller = RetrievalController(repository=repository, strategies=list(router.strategies_for(decision)))
            response = controller.retrieve(self.semantic_runner._request(record))
            traces.append(self._trace_record(record, response.results, decision.route_name, 0.0))
            routing_log.append({**decision.to_dict(), "category": record.get("category", "")})
        return traces, routing_log

    def _fusion_weight_search(self, records: list[dict[str, Any]], repository: Any, graph_repository: Any) -> dict[str, Any]:
        candidates = [
            {"dense": 0.40, "sparse": 0.30, "temporal": 0.15, "graph": 0.15},
            {"dense": 0.20, "sparse": 0.55, "temporal": 0.15, "graph": 0.10},
            {"dense": 0.15, "sparse": 0.65, "temporal": 0.10, "graph": 0.10},
            {"dense": 0.20, "sparse": 0.35, "temporal": 0.30, "graph": 0.15},
            {"dense": 0.15, "sparse": 0.35, "temporal": 0.10, "graph": 0.40},
        ]
        search_results = []
        best_trace: list[dict[str, Any]] = []
        best_weights = candidates[0]
        best_score = -1.0
        for weights in candidates:
            trace = self._trace_with_weights(records, repository, graph_repository, weights)
            metrics = summarize_retrieval_metrics(trace)
            score = metrics["recall_at_5"] + metrics["mrr"]
            search_results.append({"weights": weights, "metrics": metrics})
            if score > best_score:
                best_score = score
                best_weights = weights
                best_trace = trace
        return {"best_weights": best_weights, "best_metrics": summarize_retrieval_metrics(best_trace), "search_results": search_results, "best_trace": best_trace}

    def _trace_record(self, record: dict[str, Any], results: tuple[RetrievalResult, ...], strategy: str, latency_ms: float) -> dict[str, Any]:
        retrieved_ids = [result.note.id for result in results]
        expected_ids = [str(item) for item in record["expected_memory_ids"]]
        rank_positions = [retrieved_ids.index(expected_id) + 1 for expected_id in expected_ids if expected_id in retrieved_ids]
        return {
            "query": str(record["query"]),
            "expected_memory_ids": expected_ids,
            "retrieved_memory_ids": retrieved_ids,
            "rank_positions": rank_positions,
            "hit": bool(rank_positions),
            "latency_ms": latency_ms,
            "retrieval_strategy": strategy,
            "category": str(record.get("category", "")),
            "strategy_scores": [
                {"memory_id": result.note.id, "score": result.score, "strategy_scores": dict(result.strategy_scores)}
                for result in results
            ],
        }

    def _failure_analysis(self, trace: list[dict[str, Any]]) -> dict[str, Any]:
        failures = [record for record in trace if not record["hit"]]
        clustered = []
        counts: Counter[str] = Counter()
        for failure in failures:
            cause = self._failure_cause(failure)
            counts[cause] += 1
            clustered.append({**failure, "cluster": cause})
        return {"total_failures": len(failures), "cluster_counts": dict(counts), "failures": clustered}

    def _failure_cause(self, trace: dict[str, Any]) -> str:
        category = trace.get("category")
        scores = trace.get("strategy_scores", [])
        if category == "graph":
            return "Graph traversal failure"
        if category == "temporal":
            return "Wrong temporal ordering"
        if category == "multihop":
            return "Graph traversal failure"
        if not any(item["strategy_scores"].get("sparse", 0.0) for item in scores):
            return "Sparse keyword miss"
        if not any(item["strategy_scores"].get("dense", 0.0) for item in scores):
            return "Dense embedding miss"
        return "Fusion ranking error"

    def _embedding_diagnostics(self, records: list[dict[str, Any]], repository: Any, trace: list[dict[str, Any]]) -> dict[str, Any]:
        diagnostics = []
        nearest_hits = 0
        all_notes = repository.list()
        for record, trace_record in zip(records, trace, strict=True):
            query_vector = np.array(stable_embedding(str(record["query"])), dtype="float32")
            expected_id = str(record["expected_memory_ids"][0])
            expected_note = repository.get(expected_id)
            expected_similarity = self._cosine(query_vector, np.array(expected_note.embedding, dtype="float32")) if expected_note else 0.0
            retrieved_note = repository.get(trace_record["retrieved_memory_ids"][0]) if trace_record["retrieved_memory_ids"] else None
            retrieved_similarity = self._cosine(query_vector, np.array(retrieved_note.embedding, dtype="float32")) if retrieved_note else 0.0
            nearest = sorted(
                ((note.id, self._cosine(query_vector, np.array(note.embedding, dtype="float32"))) for note in all_notes),
                key=lambda item: item[1],
                reverse=True,
            )
            nearest_rank = next((index for index, item in enumerate(nearest, 1) if item[0] == expected_id), 0)
            if nearest_rank == 1:
                nearest_hits += 1
            diagnostics.append(
                {
                    "query": record["query"],
                    "expected_memory_id": expected_id,
                    "expected_similarity": expected_similarity,
                    "top_retrieved_similarity": retrieved_similarity,
                    "embedding_margin": round(expected_similarity - retrieved_similarity, 6),
                    "nearest_neighbor_rank": nearest_rank,
                }
            )
        margins = [item["embedding_margin"] for item in diagnostics]
        return {
            "nearest_neighbor_accuracy": round(nearest_hits / len(records), 6),
            "average_expected_similarity": self._avg(item["expected_similarity"] for item in diagnostics),
            "average_top_retrieved_similarity": self._avg(item["top_retrieved_similarity"] for item in diagnostics),
            "average_embedding_margin": self._avg(margins),
            "diagnostics": diagnostics,
        }

    def _graph_diagnostics(self, records: list[dict[str, Any]], graph_repository: Any, trace: list[dict[str, Any]]) -> dict[str, Any]:
        graph_records = [item for item in trace if item["category"] == "graph"]
        diagnostics = []
        for item in graph_records:
            graph_scores = [score for result in item["strategy_scores"] for score in [result["strategy_scores"].get("graph", 0.0)] if score]
            best_rank = min(item["rank_positions"]) if item["rank_positions"] else None
            diagnostics.append(
                {
                    "query": item["query"],
                    "graph_used": bool(graph_scores),
                    "graph_nodes_visited": len(graph_repository.nodes),
                    "graph_edges_traversed": len(graph_repository.edges),
                    "graph_score": max(graph_scores) if graph_scores else 0.0,
                    "final_rank": best_rank,
                }
            )
        return {
            "graph_query_count": len(graph_records),
            "graph_recall": summarize_retrieval_metrics(graph_records),
            "diagnostics": diagnostics,
        }

    def _temporal_diagnostics(self, records: list[dict[str, Any]], trace: list[dict[str, Any]]) -> dict[str, Any]:
        temporal_records = [item for item in trace if item["category"] == "temporal"]
        return {
            "temporal_query_count": len(temporal_records),
            "temporal_metrics": summarize_retrieval_metrics(temporal_records),
            "ordering_failures": [item for item in temporal_records if not item["hit"]],
        }

    def _reranker_analysis(self, records: list[dict[str, Any]], repository: Any, graph_repository: Any) -> dict[str, Any]:
        weights = {"dense": 0.20, "sparse": 0.55, "temporal": 0.15, "graph": 0.10}
        dense_config = SemanticRetrievalConfiguration("dense", self.semantic_runner._configurations(graph_repository)[0].strategies)
        dense_trace = self.semantic_runner._run_configuration(dense_config, repository, records)
        hybrid_trace = self._trace_with_weights(records, repository, graph_repository, weights, reranker=PassthroughReranker())
        reranked_trace = self._trace_with_weights(records, repository, graph_repository, weights)
        return {
            "dense_ranking": summarize_retrieval_metrics(dense_trace),
            "hybrid_ranking": summarize_retrieval_metrics(hybrid_trace),
            "reranked_ranking": summarize_retrieval_metrics(reranked_trace),
        }

    def _hard_query_analysis(self, trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
        analysis = []
        for item in trace:
            best_rank = min(item["rank_positions"]) if item["rank_positions"] else None
            if best_rank == 1:
                continue
            recommendation = "Improve semantic representation or add stronger entity matching."
            if item["category"] == "graph":
                recommendation = "Inspect graph edge creation and seed selection for graph route."
            elif item["category"] == "temporal":
                recommendation = "Tune temporal decay so recent relevant notes outrank generic recent notes."
            elif item["category"] == "multihop":
                recommendation = "Improve multi-hop graph expansion and second-hop scoring."
            analysis.append({**item, "hard_query_type": self._hard_type(item), "recommendation": recommendation})
        return analysis

    def _hard_type(self, item: dict[str, Any]) -> str:
        category = item["category"]
        if category == "graph":
            return "graph failures"
        if category == "temporal":
            return "temporal failures"
        if category == "multihop":
            return "multi-hop failures"
        if item["hit"]:
            return "semantic paraphrases"
        return "entity ambiguity"

    def _optimization_results(
        self,
        baseline_trace: list[dict[str, Any]],
        optimized_trace: list[dict[str, Any]],
        routed_trace: list[dict[str, Any]],
        best_trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "phase_4_1_current": summarize_retrieval_metrics(baseline_trace),
            "optimized_weighted": summarize_retrieval_metrics(optimized_trace),
            "adaptive_routed": summarize_retrieval_metrics(routed_trace),
            "best_weight_search": summarize_retrieval_metrics(best_trace),
            "comparison": {
                "recall_at_1_gain": round(summarize_retrieval_metrics(best_trace)["recall_at_1"] - summarize_retrieval_metrics(baseline_trace)["recall_at_1"], 6),
                "recall_at_5_gain": round(summarize_retrieval_metrics(best_trace)["recall_at_5"] - summarize_retrieval_metrics(baseline_trace)["recall_at_5"], 6),
                "mrr_gain": round(summarize_retrieval_metrics(best_trace)["mrr"] - summarize_retrieval_metrics(baseline_trace)["mrr"], 6),
                "ndcg_at_5_gain": round(summarize_retrieval_metrics(best_trace)["ndcg_at_5"] - summarize_retrieval_metrics(baseline_trace)["ndcg_at_5"], 6),
            },
        }

    def _component_contribution(self, records: list[dict[str, Any]], repository: Any, graph_repository: Any, weights: dict[str, float]) -> dict[str, Any]:
        configs = self.semantic_runner._configurations(graph_repository)
        traces = {
            "dense_baseline": self.semantic_runner._run_configuration(configs[0], repository, records),
            "dense_plus_sparse": self.semantic_runner._run_configuration(configs[1], repository, records),
            "dense_plus_sparse_plus_temporal": self.semantic_runner._run_configuration(configs[2], repository, records),
            "final_retrieval_stack": self._trace_with_weights(records, repository, graph_repository, weights),
        }
        metrics = {name: summarize_retrieval_metrics(trace) for name, trace in traces.items()}
        return {
            "metrics": metrics,
            "gains": {
                "sparse_gain": round(metrics["dense_plus_sparse"]["recall_at_5"] - metrics["dense_baseline"]["recall_at_5"], 6),
                "temporal_gain": round(metrics["dense_plus_sparse_plus_temporal"]["recall_at_5"] - metrics["dense_plus_sparse"]["recall_at_5"], 6),
                "graph_gain": round(metrics["final_retrieval_stack"]["recall_at_5"] - metrics["dense_plus_sparse_plus_temporal"]["recall_at_5"], 6),
                "total_gain": round(metrics["final_retrieval_stack"]["recall_at_5"] - metrics["dense_baseline"]["recall_at_5"], 6),
            },
        }

    def _cosine(self, left: np.ndarray, right: np.ndarray) -> float:
        denominator = float((np.linalg.norm(left) or 1.0) * (np.linalg.norm(right) or 1.0))
        return round(float(np.dot(left, right) / denominator), 6)

    def _avg(self, values: Any) -> float:
        values_list = [float(value) for value in values]
        return round(sum(values_list) / len(values_list), 6) if values_list else 0.0

    def _write(self, filename: str, payload: Any) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_retrieval_optimization() -> dict[str, Any]:
    """Run Phase 4.2 retrieval optimization diagnostics."""
    return RetrievalOptimizationRunner().run()


if __name__ == "__main__":
    run_retrieval_optimization()
