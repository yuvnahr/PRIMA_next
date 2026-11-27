"""Sandboxed registered-tool executor."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

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

    _pending: set[asyncio.Future[ToolResult]] = field(default_factory=set, init=False)

    def _observe(self, task: asyncio.Future[ToolResult]) -> None:
        self._pending.discard(task)
        if not task.cancelled():
            task.exception()

    async def drain(self, timeout: float = 1.0) -> bool:
        """Wait for outstanding cooperative handlers; False requires reconciliation."""
        if not self._pending:
            return True
        _, pending = await asyncio.wait(tuple(self._pending), timeout=timeout)
        return not pending

    async def execute(
        self,
        invocation: ToolInvocation,
        policy: ExecutionPolicy,
        invocation_index: int = 0,
    ) -> ToolResult:
        """Execute one invocation after policy and schema validation."""
        audit = [self._audit("tool_invocation_received", invocation.tool_name, invocation.invocation_id)]
        policy_decision = policy.evaluate_invocation(invocation, invocation_index)
        audit.append(
            self._audit(
                "policy_decision",
                policy_decision.reason,
                invocation.invocation_id,
                policy_decision.to_dict(),
            )
        )
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
        unsupported_tags = set(invocation.sandbox_tags) - set(tool.sandbox_tags)
        if unsupported_tags:
            audit.append(
                self._audit(
                    "tool_sandbox_tags_rejected",
                    "Invocation requested sandbox tags unsupported by the registered tool.",
                    invocation.invocation_id,
                    {"unsupported_tags": sorted(unsupported_tags)},
                )
            )
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=invocation.tool_name,
                status=ToolExecutionStatus.BLOCKED,
                error="Invocation requested sandbox tags unsupported by the registered tool.",
                audit=tuple(audit),
            )

        validation = self.validator.validate(invocation, tool.schema)
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
        try:
            canonical = json.dumps(validation.sanitized_arguments, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (ValueError, RecursionError) as exc:
            return ToolResult(invocation.invocation_id, invocation.tool_name, ToolExecutionStatus.VALIDATION_ERROR,
                              error=str(exc), audit=tuple(audit))
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        audit.append(self._audit("validated_operation", invocation.tool_name, invocation.invocation_id,
                                 {**invocation.metadata, "arguments_sha256": fingerprint, "retry_safe": False}))
        invocation = ToolInvocation(
            tool_name=invocation.tool_name,
            arguments=validation.sanitized_arguments,
            invocation_id=invocation.invocation_id,
            action_id=invocation.action_id,
            requested_by=invocation.requested_by,
            invocation_kind=invocation.invocation_kind,
            sandboxed=invocation.sandboxed,
            sandbox_tags=invocation.sandbox_tags,
            metadata=invocation.metadata,
        )
        if not policy.require_validation:
            audit.append(
                self._audit(
                    "validation_forced",
                    "Validation is enforced even when policy.require_validation is false.",
                    invocation.invocation_id,
                )
            )

        try:
            timeout = policy.tool_timeout_seconds
            if timeout:
                if (
                    not tool.cooperative_deadline
                    or not inspect.iscoroutinefunction(tool.handler.execute)
                    or len(self._pending) >= 32
                ):
                    return ToolResult(
                        invocation.invocation_id,
                        invocation.tool_name,
                        ToolExecutionStatus.BLOCKED,
                        error="Cooperative deadline handler unavailable or at capacity.",
                        audit=tuple(audit),
                    )
                loop = asyncio.get_running_loop()
                deadline = loop.time() + timeout
                invocation = replace(invocation, metadata={**invocation.metadata, "deadline_monotonic": deadline})
                task = asyncio.ensure_future(tool.handler.execute(invocation))
                self._pending.add(task)
                task.add_done_callback(self._observe)
                try:
                    done, _ = await asyncio.wait((task,), timeout=max(0.0, deadline - loop.time()))
                except asyncio.CancelledError:
                    task.cancel()
                    raise
                if not done or loop.time() >= deadline:
                    task.cancel()
                    raise TimeoutError
                result = task.result()
            else:
                value = tool.handler.execute(invocation)
                result = await value if inspect.isawaitable(value) else value
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
                invocation_id=invocation.invocation_id,
                tool_name=invocation.tool_name,
                status=result.status,
                output=result.output,
                error=result.error,
                audit=tuple((*audit, *result.audit)),
                metadata={**result.metadata, "arguments_sha256": fingerprint, "retry_safe": False},
            )
        except TimeoutError:
            audit.append(self._audit("tool_execution_timeout", invocation.tool_name, invocation.invocation_id))
            return ToolResult(
                invocation_id=invocation.invocation_id,
                tool_name=invocation.tool_name,
                status=ToolExecutionStatus.FAILED,
                error="Tool execution timed out. Completion is unknown.",
                metadata={"completion": "unknown", "retry_safe": False},
                audit=tuple(audit),
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

    def _audit(
        self,
        event: str,
        message: str,
        invocation_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "event": event,
            "message": message,
            "invocation_id": invocation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
