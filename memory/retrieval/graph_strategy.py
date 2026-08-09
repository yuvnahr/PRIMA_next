"""Graph traversal retrieval strategy."""

from __future__ import annotations

from memory.graph.graph_reasoning_engine import GraphReasoningEngine
from memory.graph.graph_repository import GraphRepository
from memory.graph.graph_traversal import GraphTraversal
from memory.memory_repository import MemoryRepository
from memory.retrieval.dense_strategy import DenseRetrievalStrategy
from memory.retrieval.retrieval_request import RetrievalRequest
from memory.retrieval.retrieval_result import RetrievalResult
from memory.retrieval.retrieval_strategy import RetrievalStrategy
from memory.retrieval.sparse_strategy import SparseRetrievalStrategy


class GraphTraversalStrategy(RetrievalStrategy):
    name = "graph"

    def __init__(
        self,
        graph_repository: GraphRepository,
        seed_strategy: DenseRetrievalStrategy | None = None,
        sparse_seed_strategy: SparseRetrievalStrategy | None = None,
        reasoning_engine: GraphReasoningEngine | None = None,
    ) -> None:
        self.graph_repository = graph_repository
        self.seed_strategy = seed_strategy or DenseRetrievalStrategy()
        self.sparse_seed_strategy = sparse_seed_strategy or SparseRetrievalStrategy()
        self.traversal = GraphTraversal(graph_repository)
        self.reasoning_engine = reasoning_engine or GraphReasoningEngine(graph_repository)

    def retrieve(self, request: RetrievalRequest, repository: MemoryRepository) -> list[RetrievalResult]:
        centrality = self.reasoning_engine.centrality_scores()
        seed_limit = max(1, min(6, request.top_k))
        seed_by_id = {}
        for seed in self.seed_strategy.retrieve(request, repository)[:seed_limit]:
            seed_by_id[seed.note.id] = seed
        for seed in self.sparse_seed_strategy.retrieve(request, repository)[:seed_limit]:
            existing = seed_by_id.get(seed.note.id)
            if existing is None or seed.score > existing.score:
                seed_by_id[seed.note.id] = seed
        seeds = sorted(seed_by_id.values(), key=lambda item: item.score, reverse=True)[:seed_limit]
        note_scores: dict[str, RetrievalResult] = {}
        for seed in seeds:
            seed_node = self.graph_repository.find_by_memory_id(seed.note.id)
            if seed_node is None:
                continue
            neighborhood = self.traversal.weighted_neighborhood(seed_node.id, max_depth=2)
            for node_id, graph_score in neighborhood.items():
                node = self.graph_repository.nodes[node_id]
                note = repository.get(node.memory_id)
                if note is None:
                    continue
                score = min(1.0, graph_score * seed.score)
                score = min(1.0, score * 0.9 + centrality.get(node_id, 0.0) * 0.1)
                if score > note_scores.get(note.id, RetrievalResult(note, 0.0)).score:
                    note_scores[note.id] = RetrievalResult(
                        note=note,
                        score=score,
                        strategy_scores={self.name: score},
                        explanation={
                            "seed": seed.note.id,
                            "graph_node": node_id,
                            "graph_reasoning": "centrality",
                            "centrality": centrality.get(node_id, 0.0),
                        },
                    )
        return sorted(note_scores.values(), key=lambda item: item.score, reverse=True)[: request.top_k]
