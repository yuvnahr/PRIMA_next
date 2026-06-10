"""Typed constants for unified confidence estimation."""

from __future__ import annotations

from enum import Enum


class ConfidenceSource(str, Enum):
    """Subsystem sources that may contribute probabilistic confidence."""

    RETRIEVAL = "retrieval"
    REFLECTION = "reflection"
    AFFECT = "affect"
    PLANNER = "planner"
    STATE = "state"
    POLICY = "policy"


class UncertaintyBand(str, Enum):
    """Human-readable uncertainty bands backed by continuous scores."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DecisionType(str, Enum):
    """Probabilistic workflow guidance emitted by the uncertainty layer."""

    CONTINUE = "continue"
    REFLECT = "reflect"
    RETRY = "retry"
    ASK_USER = "ask_user"


class ConfidenceTrend(str, Enum):
    """Trend direction for confidence updates."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"
