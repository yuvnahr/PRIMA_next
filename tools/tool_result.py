"""Tool execution result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolExecutionStatus(str, Enum):
    """Tool execution status."""

    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"
    BLOCKED = "blocked"
    NOT_FOUND = "not_found"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured result returned by the tool layer."""

    invocation_id: str
    tool_name: str
    status: ToolExecutionStatus
    output: Any | None = None
    error: str | None = None
    audit: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ToolExecutionStatus(self.status))
        object.__setattr__(self, "audit", tuple(dict(item) for item in self.audit))

    @property
    def succeeded(self) -> bool:
        """Return whether the tool completed successfully."""
        return self.status == ToolExecutionStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into plain Python values."""
        return {
            "invocation_id": self.invocation_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "audit": [dict(item) for item in self.audit],
            "metadata": dict(self.metadata),
            "succeeded": self.succeeded,
        }
