"""Sandbox policy for action and tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from action.tool_invocation import ToolInvocation, ToolInvocationKind


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
    allow_environment_operations: bool = False
    require_validation: bool = True
    require_sandbox: bool = True
    max_tool_invocations: int = 3
    tool_timeout_seconds: float | None = 5.0
    allowed_sandbox_tags: tuple[str, ...] = ()
    audit_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_tools", tuple(str(item) for item in self.allowed_tools))
        object.__setattr__(self, "blocked_tools", tuple(str(item) for item in self.blocked_tools))
        object.__setattr__(self, "max_tool_invocations", max(0, int(self.max_tool_invocations)))
        object.__setattr__(self, "allowed_sandbox_tags", tuple(str(item) for item in self.allowed_sandbox_tags))
        if self.tool_timeout_seconds is not None:
            object.__setattr__(self, "tool_timeout_seconds", max(0.0, float(self.tool_timeout_seconds)))

    def evaluate_invocation(self, invocation: ToolInvocation, invocation_index: int) -> PolicyDecision:
        """Return whether a tool invocation is allowed by sandbox policy."""
        if not invocation.tool_name:
            return PolicyDecision(False, "Tool name cannot be empty.")
        if not self.allow_external_actions:
            return PolicyDecision(False, "External actions are disabled by execution policy.")
        if self.require_sandbox and not invocation.sandboxed:
            return PolicyDecision(False, "Invocation is not marked for sandboxed execution.")
        if invocation.invocation_kind == ToolInvocationKind.ENVIRONMENT_OPERATION and not self.allow_environment_operations:
            return PolicyDecision(False, "Environment operations are disabled by execution policy.")
        if invocation_index >= self.max_tool_invocations:
            return PolicyDecision(False, "Tool invocation limit exceeded.")
        if invocation.tool_name in self.blocked_tools:
            return PolicyDecision(False, f"Tool '{invocation.tool_name}' is explicitly blocked.")
        if invocation.tool_name not in self.allowed_tools:
            return PolicyDecision(False, f"Tool '{invocation.tool_name}' is not allowlisted.")
        if self.allowed_sandbox_tags:
            missing_tags = set(invocation.sandbox_tags) - set(self.allowed_sandbox_tags)
            if missing_tags:
                return PolicyDecision(
                    False,
                    "Invocation requests sandbox tags outside policy.",
                    {"missing_tags": sorted(missing_tags)},
                )
        return PolicyDecision(True, "Invocation allowed by execution policy.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy into plain Python values."""
        return {
            "allowed_tools": list(self.allowed_tools),
            "blocked_tools": list(self.blocked_tools),
            "allow_external_actions": self.allow_external_actions,
            "allow_environment_operations": self.allow_environment_operations,
            "require_validation": self.require_validation,
            "require_sandbox": self.require_sandbox,
            "max_tool_invocations": self.max_tool_invocations,
            "tool_timeout_seconds": self.tool_timeout_seconds,
            "allowed_sandbox_tags": list(self.allowed_sandbox_tags),
            "audit_enabled": self.audit_enabled,
            "metadata": dict(self.metadata),
        }
