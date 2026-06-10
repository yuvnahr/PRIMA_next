"""Sandboxed registered-tool executor."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone

from action.execution_policy import ExecutionPolicy
from action.tool_invocation import ToolInvocation
from tools.tool_registry import ToolRegistry
from tools.tool_result import ToolExecutionStatus, ToolResult
from tools.tool_router import ToolRouter
from tools.tool_validator import ToolValidator


@dataclass(slots=True)
class ToolExecutor:
    """Validate, route, audit, and execute registered tools."""

    registry: ToolRegistry = field(default_factory=ToolRegistry)
    validator: ToolValidator = field(default_factory=ToolValidator)

    async def execute(
        self,
        invocation: ToolInvocation,
        policy: ExecutionPolicy,
        invocation_index: int = 0,
    ) -> ToolResult:
        """Execute one invocation after policy and schema validation."""
        audit = [self._audit("tool_invocation_received", invocation.tool_name, invocation.invocation_id)]
        policy_decision = policy.evaluate_invocation(invocation, invocation_index)
        audit.append(self._audit("policy_decision", policy_decision.reason, invocation.invocation_id))
        if not policy_decision.allowed:
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=invocation.tool_name,
                status=ToolExecutionStatus.BLOCKED,
                error=policy_decision.reason,
                audit=tuple(audit),
            )

        tool = ToolRouter(self.registry).route(invocation)
        if tool is None:
            audit.append(self._audit("tool_route_failed", "Tool not registered.", invocation.invocation_id))
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=invocation.tool_name,
                status=ToolExecutionStatus.NOT_FOUND,
                error=f"Tool '{invocation.tool_name}' is not registered.",
                audit=tuple(audit),
            )

        validation = self.validator.validate(invocation, tool.schema) if policy.require_validation else None
        if validation is not None:
            audit.append(self._audit("validation_completed", str(validation.is_valid), invocation.invocation_id))
            if not validation.is_valid:
                return ToolResult(
                    invocation_id=invocation.invocation_id,
                    tool_name=invocation.tool_name,
                    status=ToolExecutionStatus.VALIDATION_ERROR,
                    error="; ".join(validation.errors),
                    audit=tuple(audit),
                    metadata={"validation": validation.to_dict()},
                )
            invocation = ToolInvocation(
                tool_name=invocation.tool_name,
                arguments=validation.sanitized_arguments,
                invocation_id=invocation.invocation_id,
                action_id=invocation.action_id,
                requested_by=invocation.requested_by,
                metadata=invocation.metadata,
            )

        try:
            result = tool.handler.execute(invocation)
            if inspect.isawaitable(result):
                result = await result
            audit.append(self._audit("tool_execution_completed", invocation.tool_name, invocation.invocation_id))
            if not isinstance(result, ToolResult):
                return ToolResult(
                    invocation_id=invocation.invocation_id,
                    tool_name=invocation.tool_name,
                    status=ToolExecutionStatus.FAILED,
                    error="Registered tool returned an invalid result type.",
                    audit=tuple(audit),
                )
            return ToolResult(
                invocation_id=result.invocation_id,
                tool_name=result.tool_name,
                status=result.status,
                output=result.output,
                error=result.error,
                audit=tuple((*audit, *result.audit)),
                metadata=dict(result.metadata),
            )
        except Exception as exc:
            audit.append(self._audit("tool_execution_failed", str(exc), invocation.invocation_id))
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=invocation.tool_name,
                status=ToolExecutionStatus.FAILED,
                error=str(exc),
                audit=tuple(audit),
            )

    def _audit(self, event: str, message: str, invocation_id: str) -> dict[str, str]:
        return {
            "event": event,
            "message": message,
            "invocation_id": invocation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
