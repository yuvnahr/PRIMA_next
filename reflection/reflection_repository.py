"""Reflection repository with Memory Fabric integration."""

from __future__ import annotations

from memory.memory_repository import MemoryRepository
from reflection.reflection_memory import ReflectionMemory, Rule


class ReflectionRepository:
    def __init__(self, memory_repository: MemoryRepository | None = None) -> None:
        self.memory_repository = memory_repository
        self._reflections: dict[str, ReflectionMemory] = {}
        self._rules: dict[str, Rule] = {}

    def save_reflection(self, memory: ReflectionMemory) -> ReflectionMemory:
        self._reflections[memory.reflection_id] = memory
        if self.memory_repository is not None:
            self.memory_repository.add(memory.to_memory_note())
        return memory

    def save_rule(self, rule: Rule) -> Rule:
        self._rules[rule.rule_id] = rule
        if self.memory_repository is not None:
            self.memory_repository.add(rule.to_memory_note())
        return rule

    def get_reflection(self, reflection_id: str) -> ReflectionMemory | None:
        return self._reflections.get(reflection_id)

    def get_rule(self, rule_id: str) -> Rule | None:
        return self._rules.get(rule_id)

    def reflection_history(self) -> list[ReflectionMemory]:
        return list(self._reflections.values())

    def rules(self) -> list[Rule]:
        return list(self._rules.values())
