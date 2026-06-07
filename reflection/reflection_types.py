"""Reflection enums and constants."""

from __future__ import annotations

from enum import Enum


class FailureType(str, Enum):
    TOOL_FAILURE = "tool_failure"
    MEMORY_FAILURE = "memory_failure"
    RETRIEVAL_FAILURE = "retrieval_failure"
    PLANNING_FAILURE = "planning_failure"
    REASONING_FAILURE = "reasoning_failure"
    HALLUCINATION_RISK = "hallucination_risk"
    GOAL_CONFLICT = "goal_conflict"
    STATE_CONFLICT = "state_conflict"
    EMOTIONAL_CONFLICT = "emotional_conflict"


class ReflectionSignalType(str, Enum):
    MEMORY_CONFLICT = "memory_conflict"
    HALLUCINATION_RISK = "hallucination_risk"
    STATE_INCONSISTENCY = "state_inconsistency"
    GOAL_CONFLICT = "goal_conflict"
    RETRIEVAL_AMBIGUITY = "retrieval_ambiguity"
    EMOTIONAL_DISSONANCE = "emotional_dissonance"
    PLANNING_FAILURE = "planning_failure"
    TOOL_FAILURE = "tool_failure"
    REASONING_FAILURE = "reasoning_failure"


class ReflectionSource(str, Enum):
    VERIFIER = "verifier"
    FAILURE_CLASSIFIER = "failure_classifier"
    AFFECT = "affect"
    RETRIEVAL_CONFIDENCE = "retrieval_confidence"
    STATE_AUDIT = "state_audit"
    EXPEL = "expel"
