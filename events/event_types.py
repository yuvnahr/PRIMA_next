"""Canonical PRIMA-NEXT event types."""

from __future__ import annotations

from enum import Enum


class EventType(str, Enum):
    """Event names emitted across PRIMA-NEXT subsystems."""

    MEMORY_CREATED = "memory_created"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_CONSOLIDATED = "memory_consolidated"
    REFLECTION_TRIGGERED = "reflection_triggered"
    REFLECTION_COMPLETED = "reflection_completed"
    STATE_CHANGED = "state_changed"
    PLAN_CREATED = "plan_created"
    PLAN_FAILED = "plan_failed"
    TOOL_EXECUTED = "tool_executed"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_PHASE_STARTED = "workflow_phase_started"
    WORKFLOW_PHASE_COMPLETED = "workflow_phase_completed"
    WORKFLOW_PHASE_RETRY = "workflow_phase_retry"
    WORKFLOW_PHASE_FAILED = "workflow_phase_failed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    WORKFLOW_FAILED = "workflow_failed"
    CUSTOM = "custom"
