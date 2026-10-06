from __future__ import annotations

import json

import pytest

from llm.generation_config import GenerationConfig
from llm.llm_client import LLMClient
from planning.planning_context import PlanningContext, PlanningMemory
from planning.task_planner import TaskPlanner
from reasoning.models import AnswerResult, EvidenceItem, ReasoningBudget, SufficiencyStatus
from workflow.answer_generation import AnswerGenerationController
from workflow.execution_context import ExecutionContext


@pytest.mark.parametrize("replan", [False, True])
def test_storage_ids_do_not_change_model_prompt_or_citation_resolution(replan: bool) -> None:
    controller = AnswerGenerationController(LLMClient(provider_name="ollama"))
    prompts = []
    for prefix in ("first-run", "second-run"):
        source = f"mem-{prefix}"
        context = ExecutionContext("What city is mentioned?")
        context.cognitive_state.goal_state["description"] = "Keep this literal mem-first-run text."
        context.cognitive_state.task_state["source_id"] = source
        planning = PlanningContext(
            objective=context.user_input, retrieval_confidence=0.9,
            retrieved_memories=(PlanningMemory(source, "Paris", 0.9, "procedural"),
                                PlanningMemory(f"other-{prefix}", "Other context", 0.8)),
        )
        planner = TaskPlanner()
        context.plan = planner.create_plan(planning)
        if replan:
            context.plan = planner.replan(planning, context.plan, "Check grounding.")
        original_plan = context.plan.to_dict()
        evidence = EvidenceItem(f"{source}:1", "Paris", source, "episodic", 0.9, 1, context.user_input)
        reasoning = AnswerResult("", SufficiencyStatus.SUFFICIENT, 0.9, (evidence,), 1, "sufficient", ReasoningBudget())
        prompt, labels = controller._prompt(context, reasoning, True)
        prompts.append(prompt)
        assert labels == {"E1": source}
        result = controller._parse(
            json.dumps({"answer": "Paris", "evidence": ["E1"], "insufficient_information": False}),
            context.user_input, labels, True, "ollama", "model", GenerationConfig(model="model", provider="ollama"), 0.0, {},
        )
        assert result.selected_source_ids == (source,)
        assert context.plan.to_dict() == original_plan
        assert source not in prompt.user_prompt.replace("Keep this literal mem-first-run text.", "")
        assert "Keep this literal mem-first-run text." in prompt.user_prompt
        assert '"text": "Paris"' in prompt.user_prompt
    assert prompts[0] == prompts[1]
