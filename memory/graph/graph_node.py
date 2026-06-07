"""Graph node model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    memory_id: str
    labels: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
