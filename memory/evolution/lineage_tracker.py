"""Lineage tracker for evolved memories."""

from __future__ import annotations

from memory.memory_lineage import MemoryLineage
from memory.memory_note import MemoryNote


class LineageTracker:
    def build_lineage(self, parents: list[MemoryNote]) -> MemoryLineage:
        parent_ids = tuple(parent.id for parent in parents)
        ancestor_ids = tuple(
            dict.fromkeys(
                ancestor
                for parent in parents
                for ancestor in (*parent.lineage.ancestor_ids, *parent.lineage.parent_ids)
            )
        )
        origin_ids = tuple(
            dict.fromkeys(
                origin
                for parent in parents
                for origin in (parent.lineage.origin_memory_ids or (parent.id,))
            )
        )
        generation = max((parent.lineage.generation for parent in parents), default=0) + 1
        abstraction_level = max((parent.lineage.abstraction_level for parent in parents), default=0) + 1
        return MemoryLineage(
            parent_ids=parent_ids,
            child_ids=(),
            ancestor_ids=ancestor_ids,
            generation=generation,
            abstraction_level=abstraction_level,
            origin_memory_ids=origin_ids,
        )
