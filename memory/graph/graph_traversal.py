"""Graph traversal helpers."""

from __future__ import annotations

from collections import deque

from memory.graph.graph_repository import GraphRepository


class GraphTraversal:
    def __init__(self, graph_repository: GraphRepository) -> None:
        self.graph_repository = graph_repository

    def weighted_neighborhood(self, node_id: str, max_depth: int = 2) -> dict[str, float]:
        scores: dict[str, float] = {node_id: 1.0}
        queue: deque[tuple[str, int, float]] = deque([(node_id, 0, 1.0)])
        while queue:
            current, depth, score = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor, edge in self.graph_repository.get_neighbors(current):
                next_score = score * edge.weight * (0.75 ** depth)
                if next_score > scores.get(neighbor.id, 0.0):
                    scores[neighbor.id] = next_score
                    queue.append((neighbor.id, depth + 1, next_score))
        return scores
