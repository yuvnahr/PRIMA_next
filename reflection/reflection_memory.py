"""Reflection memory and rule models."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from memory.memory_context import StateSnapshot
from memory.memory_note import MemoryNote
from memory.memory_types import MemoryLevel, MemoryType
from reflection.reflection_lineage import ReflectionLineage
from reflection.reflection_signal import ReflectionSignal


@dataclass(frozen=True, slots=True)
class ReflectionMemory:
    reflection_id: str
    source_failure: str
    reflection_text: str
    reflection_signal: ReflectionSignal
    confidence: float
    lineage: ReflectionLineage
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retrieval_context: dict[str, Any] = field(default_factory=dict)
    state_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_memory_note(self) -> MemoryNote:
        return MemoryNote.create(
            content=self.reflection_text,
            memory_type=MemoryType.SEMANTIC,
            memory_level=MemoryLevel.SEMANTIC_ABSTRACTION,
            affective_state={},
            context={
                "type": "reflection_memory",
                "source_failure": self.source_failure,
                "reflection_id": self.reflection_id,
                "retrieval_context": self.retrieval_context,
            },
            state_snapshot=StateSnapshot.from_dict(self.state_snapshot),
            salience_score=self.confidence,
            retention_score=1.0,
            note_id=f"reflection_{self.reflection_id}",
        ).with_updates(
            evolution_metadata={
                "reflection_signal": self.reflection_signal.to_dict(),
                "reflection_lineage": self.lineage.to_dict(),
            }
        )


@dataclass(frozen=True, slots=True)
class Rule:
    rule_id: str
    rule_text: str
    source_failures: tuple[str, ...]
    confidence: float
    applicability: tuple[str, ...]
    lineage: ReflectionLineage
    creation_context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        rule_text: str,
        source_failures: tuple[str, ...] = (),
        confidence: float = 1.0,
        applicability: tuple[str, ...] = (),
        lineage: ReflectionLineage | None = None,
        creation_context: dict[str, Any] | None = None,
    ) -> Rule:
        return cls(
            rule_id=f"rule_{uuid.uuid4()}",
            rule_text=rule_text,
            source_failures=source_failures,
            confidence=max(0.0, min(1.0, confidence)),
            applicability=applicability,
            lineage=lineage or ReflectionLineage(source_failure_ids=source_failures),
            creation_context=creation_context or {},
        )

    def to_memory_note(self) -> MemoryNote:
        return MemoryNote.create(
            content=self.rule_text,
            memory_type=MemoryType.SEMANTIC,
            memory_level=MemoryLevel.SEMANTIC_ABSTRACTION,
            context={
                "type": "expel_rule",
                "rule_id": self.rule_id,
                "source_failures": list(self.source_failures),
                "applicability": list(self.applicability),
            },
            salience_score=self.confidence,
            note_id=f"rule_{self.rule_id}",
        ).with_updates(evolution_metadata={"reflection_lineage": self.lineage.to_dict()})



