"""Memory fabric enums and typed constants."""

from __future__ import annotations

from enum import Enum


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    EMOTIONAL = "emotional"
    PROCEDURAL = "procedural"


class MemoryLevel(str, Enum):
    RAW = "raw"
    EPISODIC_EVENT = "episodic_event"
    SEMANTIC_ABSTRACTION = "semantic_abstraction"
    EMOTIONAL_TRACE = "emotional_trace"
    VERIFIED_PROCEDURE = "verified_procedure"


class RetrievalWindow(str, Enum):
    LAST_HOUR = "last_hour"
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LONG_TERM = "long_term"


COLLECTION_BY_TYPE = {
    MemoryType.WORKING: "working_memory",
    MemoryType.EPISODIC: "episodic_memory",
    MemoryType.SEMANTIC: "semantic_memory",
    MemoryType.EMOTIONAL: "emotional_memory",
    MemoryType.PROCEDURAL: "procedural_memory",
}
