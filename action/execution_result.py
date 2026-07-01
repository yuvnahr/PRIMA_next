"""Action execution result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from action.action_context import ActionAuditRecord


class ActionExecutionStatus(str, Enum):
    """Execution status for action-layer outputs."""

    SUCCESS = "success"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"
    NO_OP = "no_op"


@dataclass(frozen=True, slots=True)
class ExecutionStepResult:
    """Result for one planned action or tool invocation."""

    action_id: str
    status: ActionExecutionStatus
    message: str
    output: Any | None = None
    tool_result: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ActionExecutionStatus(self.status))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the step result into plain Python values."""
        tool_payload = (self.tool_result.to_dict() if (self.tool_result is not None and hasattr(self.tool_result, "to_dict")) else self.tool_result)
        return {
            "action_id": self.action_id,
            "status": self.status.value,
            "message": self.message,
            "output": self.output,
            "tool_result": tool_payload,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Aggregate action execution result."""

    status: ActionExecutionStatus
    intent_type: str
    steps: tuple[ExecutionStepResult, ...] = ()
    audit_log: tuple[ActionAuditRecord, ...] = ()
    output: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ActionExecutionStatus(self.status))
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "audit_log", tuple(self.audit_log))

    @property
    def requires_attention(self) -> bool:
        """Return whether execution did not complete cleanly."""
        return self.status in {ActionExecutionStatus.PARTIAL, ActionExecutionStatus.BLOCKED, ActionExecutionStatus.FAILED}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the execution result into plain Python values."""
        return {
            "status": self.status.value,
            "intent_type": self.intent_type,
            "requires_external_tool": bool(self.metadata.get("tool_invocation_count", 0)),
            "steps": [step.to_dict() for step in self.steps],
            "audit_log": [record.to_dict() for record in self.audit_log],
            "output": self.output,
            "metadata": dict(self.metadata),
            "requires_attention": self.requires_attention,
        }
