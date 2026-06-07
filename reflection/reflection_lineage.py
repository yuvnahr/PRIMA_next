"""Reflection lineage model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ReflectionLineage:
    parent_reflection_id: str | None = None
    child_reflection_ids: tuple[str, ...] = ()
    source_failure_ids: tuple[str, ...] = ()
    generated_rule_ids: tuple[str, ...] = ()
    generation: int = 0

    def with_child(self, child_id: str) -> "ReflectionLineage":
        return ReflectionLineage(
            parent_reflection_id=self.parent_reflection_id,
            child_reflection_ids=tuple(dict.fromkeys((*self.child_reflection_ids, child_id))),
            source_failure_ids=self.source_failure_ids,
            generated_rule_ids=self.generated_rule_ids,
            generation=self.generation,
        )

    def with_rule(self, rule_id: str) -> "ReflectionLineage":
        return ReflectionLineage(
            parent_reflection_id=self.parent_reflection_id,
            child_reflection_ids=self.child_reflection_ids,
            source_failure_ids=self.source_failure_ids,
            generated_rule_ids=tuple(dict.fromkeys((*self.generated_rule_ids, rule_id))),
            generation=self.generation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_reflection_id": self.parent_reflection_id,
            "child_reflection_ids": list(self.child_reflection_ids),
            "source_failure_ids": list(self.source_failure_ids),
            "generated_rule_ids": list(self.generated_rule_ids),
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ReflectionLineage":
        data = data or {}
        return cls(
            parent_reflection_id=data.get("parent_reflection_id"),
            child_reflection_ids=tuple(data.get("child_reflection_ids", ())),
            source_failure_ids=tuple(data.get("source_failure_ids", ())),
            generated_rule_ids=tuple(data.get("generated_rule_ids", ())),
            generation=int(data.get("generation", 0)),
        )
