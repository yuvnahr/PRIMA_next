"""Lightweight query-aware retrieval routing."""

from __future__ import annotations

from dataclasses import dataclass

from memory.graph.graph_repository import GraphRepository
from memory.retrieval.dense_strategy import DenseRetrievalStrategy
from memory.retrieval.graph_strategy import GraphTraversalStrategy
from memory.retrieval.retrieval_strategy import RetrievalStrategy
from memory.retrieval.sparse_strategy import SparseRetrievalStrategy
from memory.retrieval.temporal_strategy import TemporalRetrievalStrategy


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Selected retrieval route and explanation."""

    query: str
    route_name: str
    strategies: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "route_name": self.route_name,
            "strategies": list(self.strategies),
            "reason": self.reason,
        }


class RetrievalRouter:
    """Select retrieval strategies with deterministic query heuristics."""

    TEMPORAL_TERMS = {"recent", "today", "yesterday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "when", "last"}
    GRAPH_TERMS = {"connected", "linked", "graph", "next", "belongs", "follow-up", "second-step"}
    IDENTITY_TERMS = {"identity", "responsible", "caretaker", "personal", "fact"}
    PREFERENCE_TERMS = {"preference", "usual", "choice", "favorite", "prefer"}

    def __init__(self, graph_repository: GraphRepository | None = None) -> None:
        self.graph_repository = graph_repository

    def route(self, query: str) -> RoutingDecision:
        lowered = query.lower()
        if any(term in lowered for term in self.GRAPH_TERMS) and self.graph_repository is not None:
            return RoutingDecision(query, "graph", ("dense", "sparse", "graph"), "graph/multihop cue detected")
        if any(term in lowered for term in self.TEMPORAL_TERMS):
            return RoutingDecision(query, "temporal", ("dense", "sparse", "temporal"), "temporal cue detected")
        if any(term in lowered for term in self.IDENTITY_TERMS):
            return RoutingDecision(query, "identity", ("dense", "sparse"), "identity cue detected")
        if any(term in lowered for term in self.PREFERENCE_TERMS):
            return RoutingDecision(query, "preference", ("dense", "sparse"), "preference cue detected")
        return RoutingDecision(query, "default", ("dense", "sparse", "temporal"), "default semantic route")

    def strategies_for(self, decision: RoutingDecision) -> tuple[RetrievalStrategy, ...]:
        strategy_map: dict[str, RetrievalStrategy] = {
            "dense": DenseRetrievalStrategy(),
            "sparse": SparseRetrievalStrategy(),
            "temporal": TemporalRetrievalStrategy(),
        }
        if self.graph_repository is not None:
            strategy_map["graph"] = GraphTraversalStrategy(self.graph_repository)
        return tuple(strategy_map[name] for name in decision.strategies if name in strategy_map)
