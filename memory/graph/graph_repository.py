"""Persistent graph repository with in-memory deterministic backing."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from threading import Lock
from typing import Any

from memory.graph.graph_edge import GraphEdge
from memory.graph.graph_node import GraphNode
from memory.memory_metadata import decode_metadata, encode_metadata


class GraphRepository:
    """Stores graph metadata. Chroma graph_index can mirror these records."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self._node_order: dict[str, int] = {}
        self.edges: dict[tuple[str, str, str, str], GraphEdge] = {}
        self._relations: dict[tuple[str, ...], dict[tuple[str, str, str, str], GraphEdge]] = defaultdict(dict)
        self._neighbors: dict[str, set[str]] = defaultdict(set)
        self._ordered_neighbors: dict[str, list[str]] = {}
        # ponytail: one lock per isolated graph; split locks only if graph access becomes contended.
        self._neighbor_lock = Lock()

    def add_node(self, node: GraphNode) -> GraphNode:
        if node.id not in self._node_order:
            self._node_order[node.id] = max(self._node_order.values(), default=-1) + 1
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        with self._neighbor_lock:
            previous = self.edges.get(edge.key)
            self.edges[edge.key] = edge
            self._relations[tuple(sorted((edge.source_id, edge.target_id)))][edge.key] = edge
            self._neighbors[edge.source_id].add(edge.target_id)
            self._neighbors[edge.target_id].add(edge.source_id)
            if previous is None or previous.weight != edge.weight:
                self._ordered_neighbors.pop(edge.source_id, None)
                self._ordered_neighbors.pop(edge.target_id, None)
        return edge

    def remove_edge(self, edge: GraphEdge) -> None:
        with self._neighbor_lock:
            self.edges.pop(edge.key, None)
            pair = tuple(sorted((edge.source_id, edge.target_id)))
            relations = self._relations[pair]
            relations.pop(edge.key, None)
            if not relations:
                self._relations.pop(pair)
                self._neighbors[edge.source_id].discard(edge.target_id)
                self._neighbors[edge.target_id].discard(edge.source_id)
            self._ordered_neighbors.pop(edge.source_id, None)
            self._ordered_neighbors.pop(edge.target_id, None)

    def remove_node(self, node_id: str) -> None:
        for edge in tuple(self.edges.values()):
            if node_id in (edge.source_id, edge.target_id):
                self.remove_edge(edge)
        self.nodes.pop(node_id, None)
        self._node_order.pop(node_id, None)
        self._neighbors.pop(node_id, None)
        self._ordered_neighbors.pop(node_id, None)

    def _strongest(self, a: str, b: str) -> GraphEdge:
        return max(self._relations[tuple(sorted((a, b)))].values(), key=lambda edge: (edge.weight, edge.key))

    def get_neighbors(self, node_id: str) -> list[tuple[GraphNode, GraphEdge]]:
        with self._neighbor_lock:
            ordered = self._ordered_neighbors.get(node_id)
            if ordered is None:
                ordered = sorted(
                    self._neighbors.get(node_id, ()),
                    key=lambda neighbor_id: (
                        self._strongest(node_id, neighbor_id).weight,
                        -self._node_order[neighbor_id],
                    ),
                    reverse=True,
                )
                self._ordered_neighbors[node_id] = ordered
            neighbors = []
            for neighbor_id in ordered:
                a, b = sorted((node_id, neighbor_id))
                edge = self._strongest(a, b)
                node = self.nodes[neighbor_id]
                neighbors.append((node, edge))
            return neighbors

    def degree(self, node_id: str) -> int:
        """Count neighbors without materializing or sorting their records."""
        return len(self._neighbors.get(node_id, ()))

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

    def induced(self, node_ids: set[str]) -> GraphRepository:
        graph = GraphRepository()
        for key, node in self.nodes.items():
            if key in node_ids:
                graph.add_node(node)
        for edge in self.edges.values():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                graph.add_edge(edge)
        return graph

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
    """Graph repository mirrored into the versioned Chroma graph collection."""

    def __init__(self, path: str = "./memory_db") -> None:
        super().__init__()
        import chromadb

        self.client: Any = chromadb.PersistentClient(path=path)
        self.collection: Any = self.client.get_or_create_collection(
            name="graph_index_v2",
            metadata={"hnsw:space": "cosine"},
        )
        self._record_ids: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
        self._copy_legacy()
        self._load_existing()

    def _copy_legacy(self) -> None:
        marker = "prima_graph_copy_complete"
        if self.collection.get(ids=[marker], include=[])["ids"]:
            return
        names = {getattr(c, "name", str(c)) for c in self.client.list_collections()}
        if "graph_index" in names:
            legacy = self.client.get_collection(name="graph_index")
            for offset in range(0, legacy.count(), 256):
                rows = legacy.get(limit=256, offset=offset, include=["embeddings", "documents", "metadatas"])
                self.collection.upsert(ids=rows["ids"], documents=rows["documents"],
                                       embeddings=rows["embeddings"], metadatas=rows["metadatas"])
        self.collection.upsert(ids=[marker], documents=["legacy copy complete"], embeddings=[[0.0] * 8],
                               metadatas=[{"record_type": "migration"}])

    def add_node(self, node: GraphNode) -> GraphNode:
        if self.nodes.get(node.id) == node:
            return node
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
        return super().add_node(node)

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        if self.edges.get(edge.key) == edge:
            return edge
        record_id = "graph_edge_" + hashlib.sha256(json.dumps(edge.key).encode()).hexdigest()
        metadata = {
            "record_type": "edge",
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "relation_type": edge.relation_type,
            "weight": edge.weight,
            "metadata": edge.metadata,
        }
        self.collection.upsert(
            ids=[record_id],
            documents=[edge.relation_type],
            embeddings=[[0.0] * 8],
            metadatas=[encode_metadata(metadata)],
        )
        legacy = self._record_ids[edge.key] - {record_id}
        if legacy:
            self.collection.delete(ids=sorted(legacy))
        self._record_ids[edge.key] = {record_id}
        return super().add_edge(edge)

    def _load_existing(self) -> None:
        result = self.collection.get(include=["metadatas"])
        deferred_edges: list[dict[str, Any]] = []
        for record_id, metadata in zip(result["ids"], result.get("metadatas", []), strict=True):
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
                deferred_edges.append({**decoded, "record_id": record_id})
        for decoded in deferred_edges:
            if decoded["source_id"] in self.nodes and decoded["target_id"] in self.nodes:
                edge = super().add_edge(
                    GraphEdge(
                        source_id=str(decoded["source_id"]),
                        target_id=str(decoded["target_id"]),
                        relation_type=str(decoded["relation_type"]),
                        weight=float(decoded.get("weight", 1.0)),
                        metadata=dict(decoded.get("metadata", {})),
                    )
                )
                self._record_ids[edge.key].add(decoded["record_id"])

    def remove_edge(self, edge: GraphEdge) -> None:
        ids = self._record_ids.get(edge.key)
        if ids:
            self.collection.delete(ids=sorted(ids))
            self._record_ids.pop(edge.key)
        super().remove_edge(edge)

    def remove_node(self, node_id: str) -> None:
        super().remove_node(node_id)
        self.collection.delete(ids=[f"graph_node_{node_id}"])
