"""Sandbox policy for action and tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from action.tool_invocation import ToolInvocation


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Policy decision for an execution or tool invocation."""

    allowed: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy decision into plain Python values."""
        return {"allowed": self.allowed, "reason": self.reason, "metadata": dict(self.metadata)}


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Explicit sandbox policy for action execution.

    By default, no external actions are allowed and no tools are allowlisted.
    Callers must opt in by passing allowed tool names.
    """

    allowed_tools: tuple[str, ...] = ()
    blocked_tools: tuple[str, ...] = ()
    allow_external_actions: bool = False
    require_validation: bool = True
    max_tool_invocations: int = 3
    audit_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_tools", tuple(str(item) for item in self.allowed_tools))
        object.__setattr__(self, "blocked_tools", tuple(str(item) for item in self.blocked_tools))
        object.__setattr__(self, "max_tool_invocations", max(0, int(self.max_tool_invocations)))

    def evaluate_invocation(self, invocation: ToolInvocation, invocation_index: int) -> PolicyDecision:
        """Return whether a tool invocation is allowed by sandbox policy."""
        if not self.allow_external_actions:
            return PolicyDecision(False, "External actions are disabled by execution policy.")
        if invocation_index >= self.max_tool_invocations:
            return PolicyDecision(False, "Tool invocation limit exceeded.")
        if invocation.tool_name in self.blocked_tools:
            return PolicyDecision(False, f"Tool '{invocation.tool_name}' is explicitly blocked.")
        if invocation.tool_name not in self.allowed_tools:
            return PolicyDecision(False, f"Tool '{invocation.tool_name}' is not allowlisted.")
        return PolicyDecision(True, "Invocation allowed by execution policy.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy into plain Python values."""
        return {
            "allowed_tools": list(self.allowed_tools),
            "blocked_tools": list(self.blocked_tools),
            "allow_external_actions": self.allow_external_actions,
            "require_validation": self.require_validation,
            "max_tool_invocations": self.max_tool_invocations,
            "audit_enabled": self.audit_enabled,
            "metadata": dict(self.metadata),
        }
