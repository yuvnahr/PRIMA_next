"""Memory consolidation between logical stores."""

from __future__ import annotations

from memory.maintenance.forgetting_policy import ForgettingPolicy
from memory.maintenance.retention_policy import RetentionPolicy
from memory.memory_note import MemoryNote
from memory.memory_repository import MemoryRepository
from memory.memory_types import MemoryLevel, MemoryType


class ConsolidationEngine:
    def __init__(
        self,
        repository: MemoryRepository,
        retention_policy: RetentionPolicy | None = None,
        forgetting_policy: ForgettingPolicy | None = None,
    ) -> None:
        self.repository = repository
        self.retention_policy = retention_policy or RetentionPolicy()
        self.forgetting_policy = forgetting_policy or ForgettingPolicy()

    def run(self) -> list[MemoryNote]:
        updated: list[MemoryNote] = []
        for note in self.repository.list():
            decayed_retention = self.retention_policy.decay(note)
            maintained = self.forgetting_policy.apply(note, decayed_retention)
            if maintained.memory_type == MemoryType.WORKING and self.retention_policy.should_promote_working(maintained):
                maintained = maintained.with_updates(
                    memory_type=MemoryType.EPISODIC,
                    memory_level=MemoryLevel.EPISODIC_EVENT,
                    context={**maintained.context, "consolidated_from": MemoryType.WORKING.value},
                )
                self.repository.add(maintained)
            else:
                self.repository.update(maintained)
            updated.append(maintained)
        return updated
