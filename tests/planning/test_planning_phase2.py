import unittest

from affect.affect_engine import DynamicAffectEngine
from memory.memory_note import MemoryNote
from memory.memory_types import MemoryType
from memory.retrieval.retrieval_confidence import RetrievalConfidence
from memory.retrieval.retrieval_controller import RetrievalResponse
from memory.retrieval.retrieval_result import RetrievalResult
from planning import Plan, PlanningContext, PlanningMemory, PlanningReflectionSignal, TaskPlanner
from planning.planning_types import ActionStatus, ActionType, ExecutionIntentType, PlanStatus, ReplanReason
from state.cognitive_state import CognitiveState


class PlanningPhase2Test(unittest.TestCase):
    def planning_context(self) -> PlanningContext:
        return PlanningContext(
            objective="Answer using the sourdough memory without guessing",
            goal_state={
                "active_goal": "answer with grounded memory",
                "priority": "high",
                "constraints": ["do not invent facts"],
            },
            task_state={"task": "qa"},
            confidence_state={"confidence": 0.5},
            environment_state={"channel": "test"},
            retrieved_memories=(
                PlanningMemory(
                    memory_id="mem_sourdough",
                    content="I baked sourdough bread for the surprise party",
                    score=0.68,
                    memory_type="episodic",
                    salience_score=0.4,
                    keywords=("sourdough", "party"),
                ),
            ),
            retrieval_confidence=0.35,
            retrieval_ambiguity=0.7,
            affective_priors={"boost_recent_memories": 0.8, "boost_emotional_memories": 0.4},
            reflection_signals=(
                PlanningReflectionSignal(
                    signal_type="retrieval_ambiguity",
                    severity=0.7,
                    confidence=0.9,
                    source="test",
                    reason="Retrieved memories are close in score.",
                ),
            ),
            affective_state={"dominant_emotion": "fear", "profile_confidence": 0.8, "dissonance_score": 0.2},
        )

    def test_planner_builds_structured_plan_from_memory_state_affect_and_signals(self) -> None:
        context = self.planning_context()
        plan = TaskPlanner().create_plan(context)

        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.status, PlanStatus.EVALUATED)
        self.assertEqual(plan.goal.description, "answer with grounded memory")
        self.assertFalse(plan.execution_intent.requires_external_tool)
        self.assertEqual(plan.execution_intent.intent_type, ExecutionIntentType.RESPOND_AFTER_REFLECTION)

        action_types = {action.action_type for action in plan.actions}
        self.assertIn(ActionType.REVIEW_STATE, action_types)
        self.assertIn(ActionType.INTEGRATE_MEMORY, action_types)
        self.assertIn(ActionType.APPLY_AFFECTIVE_PRIORS, action_types)
        self.assertIn(ActionType.REQUEST_REFLECTION, action_types)
        self.assertIn(ActionType.PREPARE_RESPONSE, action_types)
        self.assertTrue(any("Planner must not execute external tools" in item.description for item in plan.constraints))
        self.assertTrue(any("Planner must not invoke language models" in item.description for item in plan.constraints))

    def test_simulation_is_pure_and_marks_reflection_checkpoint(self) -> None:
        context = self.planning_context()
        planner = TaskPlanner()
        plan = planner.create_plan(context)
        before = context.to_dict()

        simulation = planner.simulate_execution(plan, context)

        self.assertEqual(context.to_dict(), before)
        self.assertTrue(simulation.requires_reflection)
        self.assertTrue(all(step.predicted_status == ActionStatus.SIMULATED for step in simulation.steps))
        self.assertTrue(
            any(step.predicted_state_changes.get("reflection_checkpoint") == "requested" for step in simulation.steps)
        )

    def test_replanning_preserves_lineage_and_adds_correction_action(self) -> None:
        context = self.planning_context()
        planner = TaskPlanner()
        original = planner.create_plan(context)

        revised = planner.replan(context, original, ReplanReason.LOW_CONFIDENCE.value)

        self.assertEqual(revised.revision, 1)
        self.assertEqual(revised.previous_plan_id, original.plan_id)
        self.assertNotEqual(revised.plan_id, original.plan_id)
        self.assertEqual(revised.actions[0].action_type, ActionType.RESOLVE_UNCERTAINTY)
        self.assertEqual(revised.metadata["replan_reason"], ReplanReason.LOW_CONFIDENCE.value)
        self.assertTrue(any("Replan because low_confidence" in item.description for item in revised.constraints))

    def test_planning_context_extracts_real_subsystem_outputs(self) -> None:
        cognitive_state = CognitiveState(
            goal_state={"active_goal": "answer_question"},
            task_state={"task": "qa"},
            confidence_state={"confidence": 0.6},
        )
        affect_update = DynamicAffectEngine().process("I am nervous about tomorrow", cognitive_state=cognitive_state)
        note = MemoryNote.create("I studied for the exam tomorrow", memory_type=MemoryType.EPISODIC)
        retrieval_response = RetrievalResponse(
            results=(RetrievalResult(note=note, score=0.72),),
            confidence=RetrievalConfidence(0.72, 0.2, 1.0, 0.72),
        )

        context = PlanningContext.from_subsystem_outputs(
            objective="What should I remember about tomorrow?",
            cognitive_state=cognitive_state,
            retrieval_response=retrieval_response,
            affect_update=affect_update,
        )

        self.assertEqual(context.goal_state["active_goal"], "answer_question")
        self.assertEqual(context.retrieved_memories[0].content, note.content)
        self.assertEqual(context.retrieval_confidence, 0.72)
        self.assertTrue(context.affective_priors)
        self.assertIn("dominant_emotion", context.affective_state)


if __name__ == "__main__":
    unittest.main()
