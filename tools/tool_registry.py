"""Explicit registry for safe tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Protocol, runtime_checkable

from action.tool_invocation import ToolInvocation, ToolInvocationKind
from tools.tool_result import ToolResult
from tools.tool_validator import ToolParameterSpec


@runtime_checkable
class ToolHandler(Protocol):
    """Protocol implemented by registered safe tool handlers."""

    def execute(self, invocation: ToolInvocation) -> ToolResult | Awaitable[ToolResult]:
        """Execute a validated invocation."""


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """Tool metadata and handler registered with the tool layer."""

    name: str
    handler: ToolHandler
    schema: tuple[ToolParameterSpec, ...] = ()
    description: str = ""
    sandbox_tags: tuple[str, ...] = ()
    invocation_kinds: tuple[ToolInvocationKind, ...] = (ToolInvocationKind.TOOL_CALL,)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "schema", tuple(self.schema))
        object.__setattr__(self, "sandbox_tags", tuple(str(item) for item in self.sandbox_tags))
        object.__setattr__(
            self,
            "invocation_kinds",
            tuple(ToolInvocationKind(item) for item in self.invocation_kinds),
        )
        if not self.invocation_kinds:
            raise ValueError("Registered tool must support at least one invocation kind.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize public tool metadata without exposing the handler."""
        return {
            "name": self.name,
            "schema": [spec.to_dict() for spec in self.schema],
            "description": self.description,
            "sandbox_tags": list(self.sandbox_tags),
            "invocation_kinds": [kind.value for kind in self.invocation_kinds],
            "metadata": dict(self.metadata),
        }


class ToolRegistry:
    """Registry for explicitly allowed tool handlers."""

    def __init__(self, tools: tuple[RegisteredTool, ...] = ()) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: RegisteredTool) -> None:
        """Register a tool by name."""
        if not tool.name:
            raise ValueError("Tool name cannot be empty.")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> RegisteredTool | None:
        """Return a registered tool by name."""
        return self._tools.get(name)

    def require(self, name: str) -> RegisteredTool:
        """Return a registered tool or raise a descriptive KeyError."""
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' is not registered.")
        return tool

    def names(self) -> tuple[str, ...]:
        """Return registered tool names."""
        return tuple(sorted(self._tools))

    def to_dict(self) -> dict[str, Any]:
        """Serialize registered public metadata."""
        return {"tools": [tool.to_dict() for tool in self._tools.values()]}
