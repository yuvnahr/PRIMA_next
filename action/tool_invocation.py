"""Typed tool invocation requests produced by the action layer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolInvocationKind(str, Enum):
    """Supported execution categories for the action/tool layer."""

    TOOL_CALL = "tool_call"
    EXTERNAL_ACTION = "external_action"
    ENVIRONMENT_OPERATION = "environment_operation"


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """Validated-request candidate for a registered tool.

    This object is only a request. The tools layer must still route, validate,
    enforce policy, and execute it.
    """

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    invocation_id: str = field(default_factory=lambda: f"tool_inv_{uuid.uuid4()}")
    action_id: str | None = None
    requested_by: str = "action_executor"
    invocation_kind: ToolInvocationKind = ToolInvocationKind.TOOL_CALL
    sandboxed: bool = True
    sandbox_tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_name", str(self.tool_name).strip())
        object.__setattr__(self, "arguments", dict(self.arguments))
        object.__setattr__(self, "invocation_kind", ToolInvocationKind(self.invocation_kind))
        object.__setattr__(self, "sandbox_tags", tuple(str(item) for item in self.sandbox_tags))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the invocation into plain Python values."""
        return {
            "invocation_id": self.invocation_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "action_id": self.action_id,
            "requested_by": self.requested_by,
            "invocation_kind": self.invocation_kind.value,
            "sandboxed": self.sandboxed,
            "sandbox_tags": list(self.sandbox_tags),
            "metadata": dict(self.metadata),
        }
