"""Retrieval router tests."""

from memory.graph.graph_repository import GraphRepository
from memory.retrieval.retrieval_router import RetrievalRouter


def test_router_selects_temporal_route() -> None:
    decision = RetrievalRouter().route("What recent Monday update should I remember?")

    assert decision.route_name == "temporal"
    assert "temporal" in decision.strategies


def test_router_selects_graph_route_when_graph_available() -> None:
    decision = RetrievalRouter(GraphRepository()).route("Which connected explanation belongs here?")

    assert decision.route_name == "graph"
    assert "graph" in decision.strategies
