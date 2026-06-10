"""Action executor that converts plans into bounded execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from action.action_context import ActionContext
from action.execution_result import ActionExecutionStatus, ExecutionResult, ExecutionStepResult
from action.tool_invocation import ToolInvocation, ToolInvocationKind
from tools.tool_executor import ToolExecutor
from tools.tool_result import ToolExecutionStatus


def _enum_value(value: Any) -> str:
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        metadata = value.get("metadata", {})
    else:
        metadata = getattr(value, "metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple | set):
        return tuple(str(item) for item in value)
    return ()


def _invocation_kind(metadata: dict[str, Any]) -> ToolInvocationKind:
    kind = metadata.get("invocation_kind", metadata.get("execution_kind", ToolInvocationKind.TOOL_CALL))
    return ToolInvocationKind(kind)


@dataclass(slots=True)
class ActionExecutor:
    """Execute plan intents through policy-gated internal and tool actions."""

    tool_executor: ToolExecutor = field(default_factory=ToolExecutor)

    async def execute(self, context: ActionContext) -> ExecutionResult:
        """Execute a plan through bounded, audited operations."""
        context.audit("action_execution_started", "Action executor received plan.", self._plan_metadata(context.plan))
        invocations = self._tool_invocations(context.plan)
        steps: list[ExecutionStepResult] = []

        if invocations:
            for index, invocation in enumerate(invocations):
                context.audit(
                    "tool_invocation_prepared",
                    f"Prepared invocation for tool '{invocation.tool_name}'.",
                    invocation.to_dict(),
                )
                tool_result = await self.tool_executor.execute(invocation, context.policy, index)
                context.audit(
                    "tool_invocation_completed",
                    f"Tool invocation finished with status {tool_result.status.value}.",
                    tool_result.to_dict(),
                )
                steps.append(
                    ExecutionStepResult(
                        action_id=invocation.action_id or invocation.invocation_id,
                        status=self._status_from_tool(tool_result.status),
                        message=tool_result.error or "Tool invocation completed.",
                        output=tool_result.output,
                        tool_result=tool_result,
                        metadata={"tool_name": invocation.tool_name},
                    )
                )
        else:
            steps.extend(self._internal_steps(context))

        status = self._aggregate_status(tuple(steps), context.plan)
        output = self._output_payload(context, tuple(steps))
        context.audit("action_execution_completed", f"Action execution completed with status {status.value}.")
        return ExecutionResult(
            status=status,
            intent_type=self._intent_type(context.plan),
            steps=tuple(steps),
            audit_log=tuple(context.audit_log),
            output=output,
            metadata={
                "plan_id": str(getattr(context.plan, "plan_id", "")),
                "tool_invocation_count": len(invocations),
                "external_action_count": sum(
                    1 for invocation in invocations if invocation.invocation_kind == ToolInvocationKind.EXTERNAL_ACTION
                ),
                "environment_operation_count": sum(
                    1
                    for invocation in invocations
                    if invocation.invocation_kind == ToolInvocationKind.ENVIRONMENT_OPERATION
                ),
                "sandboxed": True,
                "validation_required": True,
            },
        )

    async def execute_plan(self, plan: Any, context: ActionContext | None = None) -> ExecutionResult:
        """Execute a plan, creating a default action context when needed."""
        action_context = context or ActionContext(plan=plan)
        return await self.execute(action_context)

    def _tool_invocations(self, plan: Any) -> tuple[ToolInvocation, ...]:
        invocations: list[ToolInvocation] = []
        intent = getattr(plan, "execution_intent", None)
        if intent is not None and bool(getattr(intent, "requires_external_tool", False)):
            tool_name = getattr(intent, "tool_name", None)
            if tool_name:
                intent_metadata = _metadata(intent)
                invocations.append(
                    ToolInvocation(
                        tool_name=str(tool_name),
                        arguments=_dict_value(intent_metadata.get("tool_args", {})),
                        action_id=str(getattr(plan, "plan_id", "")) or None,
                        invocation_kind=_invocation_kind(intent_metadata),
                        sandboxed=bool(intent_metadata.get("sandboxed", True)),
                        sandbox_tags=_string_tuple(intent_metadata.get("sandbox_tags", ())),
                        metadata={"source": "execution_intent"},
                    )
                )
        elif isinstance(plan, dict):
            intent_data = plan.get("execution_intent") or {}
            if isinstance(intent_data, dict) and bool(intent_data.get("requires_external_tool", False)):
                tool_name = intent_data.get("tool_name")
                intent_metadata = _dict_value(intent_data.get("metadata", {}))
                if tool_name:
                    invocations.append(
                        ToolInvocation(
                            tool_name=str(tool_name),
                            arguments=_dict_value(intent_metadata.get("tool_args", {})),
                            action_id=str(plan.get("plan_id", "")) or None,
                            invocation_kind=_invocation_kind(intent_metadata),
                            sandboxed=bool(intent_metadata.get("sandboxed", True)),
                            sandbox_tags=_string_tuple(intent_metadata.get("sandbox_tags", ())),
                            metadata={"source": "execution_intent"},
                        )
                    )

        for action in self._actions(plan):
            metadata = _metadata(action)
            tool_name = metadata.get("tool_name")
            if not tool_name:
                continue
            invocations.append(
                ToolInvocation(
                    tool_name=str(tool_name),
                    arguments=_dict_value(metadata.get("tool_args", {})),
                    action_id=self._action_id(action),
                    invocation_kind=_invocation_kind(metadata),
                    sandboxed=bool(metadata.get("sandboxed", True)),
                    sandbox_tags=_string_tuple(metadata.get("sandbox_tags", ())),
                    metadata={"source": "plan_action", "action_type": self._action_type(action)},
                )
            )
        return tuple(invocations)

    def _internal_steps(self, context: ActionContext) -> tuple[ExecutionStepResult, ...]:
        steps: list[ExecutionStepResult] = []
        actions = self._actions(context.plan)
        if not actions:
            return (
                ExecutionStepResult(
                    action_id=str(getattr(context.plan, "plan_id", "plan")),
                    status=ActionExecutionStatus.NO_OP,
                    message="Plan contains no executable actions.",
                    metadata={"sandboxed": True},
                ),
            )

        for action in actions:
            action_id = self._action_id(action)
            action_type = self._action_type(action)
            context.audit(
                "internal_action_recorded",
                f"Recorded internal action '{action_type}' without external side effects.",
                {"action_id": action_id, "action_type": action_type},
            )
            steps.append(
                ExecutionStepResult(
                    action_id=action_id,
                    status=ActionExecutionStatus.SUCCESS,
                    message="Internal action recorded; no external operation executed.",
                    output={
                        "action_type": action_type,
                        "expected_output": self._expected_output(action),
                        "external_side_effect": False,
                    },
                    metadata={"sandboxed": True},
                )
            )
        return tuple(steps)

    def _actions(self, plan: Any) -> tuple[Any, ...]:
        actions = getattr(plan, "actions", None)
        if actions is not None:
            return tuple(actions)
        if isinstance(plan, dict):
            return tuple(plan.get("actions", ()))
        return ()

    def _intent_type(self, plan: Any) -> str:
        intent = getattr(plan, "execution_intent", None)
        if intent is not None:
            return _enum_value(getattr(intent, "intent_type", "unknown"))
        if isinstance(plan, dict):
            intent_data = plan.get("execution_intent") or {}
            return str(intent_data.get("intent_type", "unknown"))
        return "unknown"

    def _action_id(self, action: Any) -> str:
        if isinstance(action, dict):
            return str(action.get("action_id", action.get("action_type", "action")))
        return str(getattr(action, "action_id", getattr(action, "action_type", "action")))

    def _action_type(self, action: Any) -> str:
        if isinstance(action, dict):
            return str(action.get("action_type", "unknown"))
        return _enum_value(getattr(action, "action_type", "unknown"))

    def _expected_output(self, action: Any) -> str:
        if isinstance(action, dict):
            return str(action.get("expected_output", ""))
        return str(getattr(action, "expected_output", ""))

    def _status_from_tool(self, status: ToolExecutionStatus) -> ActionExecutionStatus:
        if status == ToolExecutionStatus.SUCCESS:
            return ActionExecutionStatus.SUCCESS
        if status in {ToolExecutionStatus.BLOCKED, ToolExecutionStatus.VALIDATION_ERROR, ToolExecutionStatus.NOT_FOUND}:
            return ActionExecutionStatus.BLOCKED
        return ActionExecutionStatus.FAILED

    def _aggregate_status(self, steps: tuple[ExecutionStepResult, ...], plan: Any) -> ActionExecutionStatus:
        if not steps:
            return ActionExecutionStatus.NO_OP
        statuses = {step.status for step in steps}
        if statuses == {ActionExecutionStatus.SUCCESS}:
            return ActionExecutionStatus.SUCCESS
        if ActionExecutionStatus.FAILED in statuses:
            return ActionExecutionStatus.FAILED
        if ActionExecutionStatus.BLOCKED in statuses:
            return ActionExecutionStatus.BLOCKED
        if self._intent_type(plan) == "no_op":
            return ActionExecutionStatus.NO_OP
        return ActionExecutionStatus.PARTIAL

    def _output_payload(self, context: ActionContext, steps: tuple[ExecutionStepResult, ...]) -> dict[str, Any]:
        return {
            "intent_type": self._intent_type(context.plan),
            "step_count": len(steps),
            "successful_steps": sum(1 for step in steps if step.status == ActionExecutionStatus.SUCCESS),
            "blocked_steps": sum(1 for step in steps if step.status == ActionExecutionStatus.BLOCKED),
            "failed_steps": sum(1 for step in steps if step.status == ActionExecutionStatus.FAILED),
            "world_prediction_available": context.world_prediction is not None,
            "uncertainty_available": context.uncertainty is not None,
        }

    def _plan_metadata(self, plan: Any) -> dict[str, Any]:
        return {
            "plan_id": str(getattr(plan, "plan_id", "")),
            "intent_type": self._intent_type(plan),
            "action_count": len(self._actions(plan)),
        }
