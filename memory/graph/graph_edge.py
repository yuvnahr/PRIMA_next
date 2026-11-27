"""Graph edge model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str, str]:
        a, b = self.source_id, self.target_id
        if self.relation_type == "shared_context":
            a, b = sorted((a, b))
        legacy = ""
        if self.relation_type == "shared_context" and set(self.metadata) == {"overlap"}:
            legacy = "graph_builder"
        elif self.relation_type == "evolves_into" and set(self.metadata) == {"generation"}:
            legacy = "memory_evolution"
        producer = self.metadata.get("producer", self.metadata.get("source", legacy))
        return a, b, self.relation_type, str(producer)
