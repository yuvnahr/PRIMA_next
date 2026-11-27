import unittest

from action import (
    ActionContext,
    ActionExecutor,
    ExecutionPolicy,
    ToolInvocation,
    ToolInvocationKind,
)
from planning.plan import ExecutionIntent, Plan, PlanAction, PlanConstraint, PlanGoal
from planning.planning_types import ActionType, ConstraintType, ExecutionIntentType, GoalPriority
from tools import (
    RegisteredTool,
    ToolExecutionStatus,
    ToolExecutor,
    ToolParameterSpec,
    ToolRegistry,
    ToolResult,
)


class EchoTool:
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            invocation_id=invocation.invocation_id,
            tool_name=invocation.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            output={"echo": invocation.arguments["message"]},
        )


class Phase5ActionToolTest(unittest.IsolatedAsyncioTestCase):
    def plan_with_tool_metadata(
        self,
        *,
        invocation_kind: ToolInvocationKind = ToolInvocationKind.TOOL_CALL,
        arguments: dict[str, object] | None = None,
    ) -> Plan:
        action = PlanAction.create(
            ActionType.PREPARE_RESPONSE,
            "Invoke a registered echo tool through the action layer.",
            ("message",),
            "echo_result",
            metadata={
                "tool_name": "echo",
                "tool_args": arguments or {"message": "hello"},
                "invocation_kind": invocation_kind.value,
                "sandboxed": True,
                "sandbox_tags": ("read_only",),
            },
        )
        constraint = PlanConstraint.create(
            ConstraintType.EXECUTION,
            "Tool execution must remain policy gated and sandboxed.",
        )
        return Plan(
            plan_id="plan_echo",
            objective="echo safely",
            goal=PlanGoal.create("echo safely", GoalPriority.MEDIUM, 0.8, "test"),
            actions=(action,),
            execution_intent=ExecutionIntent(
                intent_type=ExecutionIntentType.RESPOND,
                objective="echo safely",
                rationale="test",
                confidence=0.8,
            ),
            constraints=(constraint,),
        )

    def executor(self) -> ActionExecutor:
        registry = ToolRegistry(
            (
                RegisteredTool(
                    name="echo",
                    handler=EchoTool(),
                    cooperative_deadline=True,
                    schema=(
                        ToolParameterSpec(
                            name="message",
                            parameter_type="str",
                            required=True,
                            allow_empty=False,
                            max_length=20,
                        ),
                    ),
                    sandbox_tags=("read_only",),
                    invocation_kinds=(ToolInvocationKind.TOOL_CALL, ToolInvocationKind.ENVIRONMENT_OPERATION),
                ),
            )
        )
        return ActionExecutor(tool_executor=ToolExecutor(registry=registry))

    async def test_default_policy_blocks_external_tool_execution(self) -> None:
        result = await self.executor().execute(ActionContext(plan=self.plan_with_tool_metadata()))

        self.assertEqual(result.status.value, "blocked")
        self.assertTrue(result.requires_attention)
        self.assertEqual(result.metadata["tool_invocation_count"], 1)
        self.assertIn("disabled", result.steps[0].message)

    async def test_allowlisted_tool_executes_after_validation(self) -> None:
        policy = ExecutionPolicy(
            allowed_tools=("echo",),
            allow_external_actions=True,
            allowed_sandbox_tags=("read_only",),
        )

        result = await self.executor().execute(ActionContext(plan=self.plan_with_tool_metadata(), policy=policy))

        self.assertEqual(result.status.value, "success")
        self.assertEqual(result.steps[0].output, {"echo": "hello"})
        self.assertEqual(result.to_dict()["requires_external_tool"], True)

    async def test_validator_rejects_unexpected_arguments(self) -> None:
        policy = ExecutionPolicy(
            allowed_tools=("echo",),
            allow_external_actions=True,
            allowed_sandbox_tags=("read_only",),
        )

        result = await self.executor().execute(
            ActionContext(
                plan=self.plan_with_tool_metadata(arguments={"message": "hello", "callable": object()}),
                policy=policy,
            )
        )

        self.assertEqual(result.status.value, "blocked")
        self.assertIn("Unexpected argument", result.steps[0].message)

    async def test_environment_operations_require_explicit_policy(self) -> None:
        policy = ExecutionPolicy(
            allowed_tools=("echo",),
            allow_external_actions=True,
            allowed_sandbox_tags=("read_only",),
        )

        result = await self.executor().execute(
            ActionContext(
                plan=self.plan_with_tool_metadata(invocation_kind=ToolInvocationKind.ENVIRONMENT_OPERATION),
                policy=policy,
            )
        )

        self.assertEqual(result.status.value, "blocked")
        self.assertEqual(result.metadata["environment_operation_count"], 1)
        self.assertIn("Environment operations are disabled", result.steps[0].message)


if __name__ == "__main__":
    unittest.main()


def test_numeric_schema_and_recursive_validation():
    import pytest

    from tools.tool_validator import ToolValidator

    for bounds in (
        {"min_value": True},
        {"max_value": float("nan")},
        {"min_value": 2, "max_value": 1},
        {"min_length": -1},
        {"max_length": 1.5},
        {"min_length": 2, "max_length": 1},
    ):
        with pytest.raises(ValueError):
            ToolParameterSpec("x", "int", **bounds)
    for kind, value, valid in (
        ("float", float("nan"), False),
        ("dict", {"a": [(float("inf"),)]}, False),
        ("int", True, False),
        ("int", 10**400, False),
        ("int", 0, True),
        ("int", 1, True),
        ("bool", True, True),
    ):
        spec = (
            ToolParameterSpec("x", kind, min_value=0, max_value=1)
            if kind in {"float", "int"}
            else ToolParameterSpec("x", kind)
        )
        result = ToolValidator().validate(ToolInvocation("tool", {"x": value}), (spec,))
        assert result.is_valid == valid


def test_bounded_tools_reject_sync_before_dispatch_and_observe_late_effect():
    import asyncio

    async def run():
        seen = []

        class Sync:
            def execute(self, invocation):
                seen.append("sync")
                return ToolResult(invocation.invocation_id, invocation.tool_name, ToolExecutionStatus.SUCCESS)

        policy = ExecutionPolicy(allowed_tools=("t",), allow_external_actions=True, tool_timeout_seconds=0.025)
        invocation = ToolInvocation("t")
        executor = ToolExecutor(ToolRegistry((RegisteredTool("t", Sync()),)))
        assert (await executor.execute(invocation, policy)).status == ToolExecutionStatus.BLOCKED
        assert seen == []

        release, cancelled = asyncio.Event(), asyncio.Event()

        class Suppresses:
            async def execute(self, invocation):
                assert invocation.metadata["deadline_monotonic"] > 0
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                    seen.append(invocation.invocation_id)
                    return ToolResult(invocation.invocation_id, "t", ToolExecutionStatus.SUCCESS)

        executor = ToolExecutor(ToolRegistry((RegisteredTool("t", Suppresses(), cooperative_deadline=True),)))
        result = await asyncio.wait_for(executor.execute(invocation, policy), 1)
        assert result.status == ToolExecutionStatus.FAILED
        assert result.metadata["completion"] == "unknown" and not result.metadata["retry_safe"]
        await asyncio.wait_for(cancelled.wait(), 1)
        assert seen == []
        release.set()
        assert await executor.drain(timeout=1)
        assert seen == [invocation.invocation_id]

    asyncio.run(run())


def test_deliberately_unbounded_handler_and_invalid_timeout_policy():
    import asyncio
    import json

    import pytest

    for value in (True, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            ExecutionPolicy(tool_timeout_seconds=value)

    class Legacy:
        def execute(self, invocation):
            json.dumps(invocation.arguments, allow_nan=False)
            return ToolResult(
                invocation.invocation_id, "t", ToolExecutionStatus.SUCCESS, output=invocation.arguments["n"]
            )

    executor = ToolExecutor(ToolRegistry((RegisteredTool("t", Legacy(), (ToolParameterSpec("n", "int"),)),)))
    for timeout in (0, None):
        policy = ExecutionPolicy(allowed_tools=("t",), allow_external_actions=True, tool_timeout_seconds=timeout)
        assert asyncio.run(executor.execute(ToolInvocation("t", {"n": 10**400}), policy)).output == 10**400
