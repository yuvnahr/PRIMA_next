"""State carried through the adaptive reflection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ReflectionState:
    query: str
    plan: str = ""
    scratchpad: str = ""
    current_proposal: str = ""
    latest_observation: str = ""
    is_verified: bool = False
    reflections: list[str] = field(default_factory=list)
    retry_count: int = 0
    final_answer: str = ""
    extracted_rule: str = ""
