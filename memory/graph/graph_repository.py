"""Persistent graph repository with in-memory deterministic backing."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from memory.graph.graph_edge import GraphEdge
from memory.graph.graph_node import GraphNode
from memory.memory_metadata import decode_metadata, encode_metadata


class GraphRepository:
    """Stores graph metadata. Chroma graph_index can mirror these records."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[tuple[str, str], GraphEdge] = {}
        self._neighbors: dict[str, set[str]] = defaultdict(set)

    def add_node(self, node: GraphNode) -> GraphNode:
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        self.edges[edge.key] = edge
        self._neighbors[edge.source_id].add(edge.target_id)
        self._neighbors[edge.target_id].add(edge.source_id)
        return edge

    def get_neighbors(self, node_id: str) -> list[tuple[GraphNode, GraphEdge]]:
        neighbors = []
        for neighbor_id in self._neighbors.get(node_id, set()):
            a, b = sorted((node_id, neighbor_id))
            edge = self.edges[(a, b)]
            node = self.nodes[neighbor_id]
            neighbors.append((node, edge))
        return sorted(neighbors, key=lambda item: item[1].weight, reverse=True)

    def find_by_memory_id(self, memory_id: str) -> GraphNode | None:
        for node in self.nodes.values():
            if node.memory_id == memory_id:
                return node
        return None

    def shortest_path(self, source_id: str, target_id: str) -> list[str]:
        if source_id == target_id:
            return [source_id]
        queue: deque[tuple[str, list[str]]] = deque([(source_id, [source_id])])
        visited = {source_id}
        while queue:
            current, path = queue.popleft()
            for neighbor in self._neighbors.get(current, set()):
                if neighbor in visited:
                    continue
                next_path = path + [neighbor]
                if neighbor == target_id:
                    return next_path
                visited.add(neighbor)
                queue.append((neighbor, next_path))
        return []

    def connected_components(self) -> list[set[str]]:
        remaining = set(self.nodes)
        components: list[set[str]] = []
        while remaining:
            seed = remaining.pop()
            component = {seed}
            queue = deque([seed])
            while queue:
                current = queue.popleft()
                for neighbor in self._neighbors.get(current, set()):
                    if neighbor not in component:
                        component.add(neighbor)
                        remaining.discard(neighbor)
                        queue.append(neighbor)
            components.append(component)
        return components


class ChromaGraphRepository(GraphRepository):
    """Graph repository mirrored into ChromaDB's `graph_index` collection."""

    def __init__(self, path: str = "./memory_db") -> None:
        super().__init__()
        import chromadb

        self.client: Any = chromadb.PersistentClient(path=path)
        self.collection: Any = self.client.get_or_create_collection(
            name="graph_index",
            metadata={"hnsw:space": "cosine"},
        )
        self._load_existing()

    def add_node(self, node: GraphNode) -> GraphNode:
        saved = super().add_node(node)
        metadata = {
            "record_type": "node",
            "node_id": node.id,
            "memory_id": node.memory_id,
            "labels": list(node.labels),
            "metadata": node.metadata,
        }
        self.collection.upsert(
            ids=[f"graph_node_{node.id}"],
            documents=[node.memory_id],
            embeddings=[[0.0] * 8],
            metadatas=[encode_metadata(metadata)],
        )
        return saved

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        saved = super().add_edge(edge)
        metadata = {
            "record_type": "edge",
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "relation_type": edge.relation_type,
            "weight": edge.weight,
            "metadata": edge.metadata,
        }
        self.collection.upsert(
            ids=[f"graph_edge_{edge.source_id}_{edge.target_id}"],
            documents=[edge.relation_type],
            embeddings=[[0.0] * 8],
            metadatas=[encode_metadata(metadata)],
        )
        return saved

    def _load_existing(self) -> None:
        try:
            result = self.collection.get(include=["metadatas"])
        except Exception:
            return
        deferred_edges: list[dict[str, Any]] = []
        for metadata in result.get("metadatas", []):
            decoded = decode_metadata(metadata)
            if decoded.get("record_type") == "node":
                super().add_node(
                    GraphNode(
                        id=str(decoded["node_id"]),
                        memory_id=str(decoded["memory_id"]),
                        labels=tuple(decoded.get("labels", ())),
                        metadata=dict(decoded.get("metadata", {})),
                    )
                )
            elif decoded.get("record_type") == "edge":
                deferred_edges.append(decoded)
        for decoded in deferred_edges:
            if decoded["source_id"] in self.nodes and decoded["target_id"] in self.nodes:
                super().add_edge(
                    GraphEdge(
                        source_id=str(decoded["source_id"]),
                        target_id=str(decoded["target_id"]),
                        relation_type=str(decoded["relation_type"]),
                        weight=float(decoded.get("weight", 1.0)),
                        metadata=dict(decoded.get("metadata", {})),
                    )
                )
