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
