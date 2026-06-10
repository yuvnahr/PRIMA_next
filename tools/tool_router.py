"""Tool routing through the explicit registry."""

from __future__ import annotations

from dataclasses import dataclass

from action.tool_invocation import ToolInvocation
from tools.tool_registry import RegisteredTool, ToolRegistry


@dataclass(slots=True)
class ToolRouter:
    """Route invocations to registered tools."""

    registry: ToolRegistry

    def route(self, invocation: ToolInvocation) -> RegisteredTool | None:
        """Return the registered tool for an invocation, if one exists."""
        if not invocation.tool_name:
            return None
        return self.registry.get(invocation.tool_name)
