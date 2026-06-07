"""Graph reasoning: communities, paths, centrality."""

from __future__ import annotations

from memory.graph.graph_repository import GraphRepository


class GraphReasoningEngine:
    def __init__(self, graph_repository: GraphRepository) -> None:
        self.graph_repository = graph_repository

    def centrality_scores(self) -> dict[str, float]:
        total_nodes = max(1, len(self.graph_repository.nodes) - 1)
        return {
            node_id: round(len(self.graph_repository.get_neighbors(node_id)) / total_nodes, 6)
            for node_id in self.graph_repository.nodes
        }

    def detect_communities(self) -> list[set[str]]:
        try:
            import networkx as nx
            from networkx.algorithms.community import louvain_communities

            graph = nx.Graph()
            for node_id in self.graph_repository.nodes:
                graph.add_node(node_id)
            for edge in self.graph_repository.edges.values():
                graph.add_edge(edge.source_id, edge.target_id, weight=edge.weight)
            if graph.number_of_edges() == 0:
                return [{node_id} for node_id in graph.nodes]
            return [set(community) for community in louvain_communities(graph, weight="weight", seed=42)]
        except Exception:
            return self.graph_repository.connected_components()

    def path_search(self, source_id: str, target_id: str) -> list[str]:
        return self.graph_repository.shortest_path(source_id, target_id)
