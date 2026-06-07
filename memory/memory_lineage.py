"""Explicit memory lineage tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MemoryLineage:
    parent_ids: tuple[str, ...] = ()
    child_ids: tuple[str, ...] = ()
    ancestor_ids: tuple[str, ...] = ()
    generation: int = 0
    abstraction_level: int = 0
    origin_memory_ids: tuple[str, ...] = ()

    def link_child(self, child_id: str) -> "MemoryLineage":
        children = tuple(dict.fromkeys((*self.child_ids, child_id)))
        return MemoryLineage(
            parent_ids=self.parent_ids,
            child_ids=children,
            ancestor_ids=self.ancestor_ids,
            generation=self.generation,
            abstraction_level=self.abstraction_level,
            origin_memory_ids=self.origin_memory_ids,
        )

    def next_generation(self, new_parent_ids: tuple[str, ...]) -> "MemoryLineage":
        ancestors = tuple(dict.fromkeys((*self.ancestor_ids, *self.parent_ids, *new_parent_ids)))
        origins = self.origin_memory_ids or new_parent_ids
        return MemoryLineage(
            parent_ids=new_parent_ids,
            child_ids=(),
            ancestor_ids=ancestors,
            generation=self.generation + 1,
            abstraction_level=self.abstraction_level + 1,
            origin_memory_ids=tuple(dict.fromkeys(origins)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_ids": list(self.parent_ids),
            "child_ids": list(self.child_ids),
            "ancestor_ids": list(self.ancestor_ids),
            "generation": self.generation,
            "abstraction_level": self.abstraction_level,
            "origin_memory_ids": list(self.origin_memory_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MemoryLineage":
        data = data or {}
        return cls(
            parent_ids=tuple(data.get("parent_ids", ())),
            child_ids=tuple(data.get("child_ids", ())),
            ancestor_ids=tuple(data.get("ancestor_ids", ())),
            generation=int(data.get("generation", 0)),
            abstraction_level=int(data.get("abstraction_level", 0)),
            origin_memory_ids=tuple(data.get("origin_memory_ids", ())),
        )
