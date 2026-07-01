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
    def key(self) -> tuple[str, str]:
        a, b = sorted((self.source_id, self.target_id))
        return (a, b)
