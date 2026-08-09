# Phase 06 — Bounded Correction and Reflection Loop

## Outcome

Reflection is now an executable, workflow-owned correction mechanism rather than a passive checkpoint. Typed advice is validated for confidence, duplication, evaluation leakage and explicit budgets before the orchestration engine routes it back to retrieval, planning, action execution or answer generation. Post-execution validation can request bounded correction, clarification or abstention. Compatibility results now use real attempt snapshots rather than copying the final answer into both before/after fields.

## Contract and behavior changes

- `ReflectionAdvice` now exposes `trigger`, bounded `action`, `suggested_query`, `correction_proposal`, `confidence`, `provenance`, `reason_code`, and the originating event.
- Supported actions are `revise_query`, `broaden_query`, `pivot_entity`, `replan`, `retry_transient_failure`, `regenerate`, `ask_user`, `abstain`, and `no_action`.
- `ReasoningReflectionAdapter` maps low-confidence, contradiction, duplicate/no-progress, plan, schema, grounding, tool, logical and policy events to usable actions accepted by the reasoning controller.
- `CorrectionBudget` independently limits retrieval retries, replans, reflection interventions and generations. The workflow also respects the Phase 05 uncertainty-gate retry limit.
- `CorrectionLoop` rejects leakage, low-confidence, duplicate and over-budget advice before it can change execution.
- `CorrectionAttempt` records schema-versioned before/after query, plan and answer snapshots, confidence delta, decision reason and utility.
- Full-profile pre-execution advice can re-enter retrieval or planning. Post-execution advice can re-enter retrieval/planning/execution, regenerate, ask the user or abstain.
- `OUTPUT_VALIDATION` detects schema failure, unsupported evidence, tool failure, logical inconsistency and policy violation without generating answers itself.
- Runtime output and diagnostics expose correction budgets and every workflow attempt. Reasoning-internal reflection attempts are retained separately in answer diagnostics.
- `RuntimeResult.prediction_before_reflection` and `prediction_after_reflection` now come from actual attempts.

## Changed files

- Advice and reasoning boundary: `reasoning/reflection_advisor.py`, `reasoning/controller.py`, `reasoning/__init__.py`, `reasoning/validate_reflection_integration.py`
- Reflection adapter/result: `reflection/reasoning_reflection_adapter.py`, `reflection/reflection_result.py`
- Correction state machine: `workflow/correction_loop.py`, `workflow/output_validation.py`, `workflow/orchestration_engine.py`, `workflow/prima_workflow.py`, `workflow/task_router.py`, `workflow/workflow_state.py`, `workflow/execution_context.py`, `workflow/__init__.py`
- Runtime result/diagnostics: `runtime/contracts.py`, `runtime/prima_runtime.py`
- Tests: `tests/workflow/test_phase06_correction_loop.py`, `tests/reasoning/test_reflection_boundary.py`
- Architecture ledger: `docs/implementation/ARCHITECTURE_TRUTH_MATRIX.md`, `docs/implementation/STATUS.md`

## Verification

```text
Focused reasoning/reflection regression:
venv\Scripts\python.exe -m pytest -q tests/reasoning/test_reflection_boundary.py tests/reasoning/test_controller.py tests/reasoning/test_controller_boundaries.py
13 passed in 10.75s.

Initial Phase 06 focused suite:
venv\Scripts\python.exe -m pytest -q tests/workflow/test_phase06_correction_loop.py tests/reasoning/test_reflection_boundary.py
6 passed in 7.73s.

Runtime/canonical regression selection after integration:
venv\Scripts\python.exe -m pytest -q tests/workflow/test_phase06_correction_loop.py tests/reasoning/test_reflection_boundary.py tests/integration/test_runtime_pipeline.py tests/runtime/test_canonical_contract.py tests/workflow/test_phase05_confidence_branch.py
28 passed, 1 failed in 7.96s. The new compatibility assertion exposed a fallback that still copied the final answer when the real pre-correction answer was empty.

Focused compatibility rerun after removing that fallback:
venv\Scripts\python.exe -m pytest -q tests/workflow/test_phase06_correction_loop.py::test_runtime_result_uses_actual_pre_and_post_correction_snapshots
1 passed in 7.05s.

Expanded Phase 06 focused suite:
venv\Scripts\python.exe -m pytest -q tests/workflow/test_phase06_correction_loop.py tests/reasoning/test_reflection_boundary.py
8 passed in 7.42s.

Consolidated Phase 05/06 and runtime regression selection:
venv\Scripts\python.exe -m pytest -q tests/workflow/test_phase06_correction_loop.py tests/reasoning/test_reflection_boundary.py tests/reasoning/test_controller.py tests/reasoning/test_controller_boundaries.py tests/integration/test_runtime_pipeline.py tests/runtime/test_canonical_contract.py tests/workflow/test_phase05_confidence_branch.py
41 passed in 8.54s.

Focused Ruff:
venv\Scripts\python.exe -m ruff check reasoning/reflection_advisor.py reasoning/controller.py reasoning/validate_reflection_integration.py reflection/reasoning_reflection_adapter.py reflection/reflection_result.py runtime/contracts.py runtime/prima_runtime.py tests/reasoning/test_reflection_boundary.py tests/workflow/test_phase06_correction_loop.py workflow/__init__.py workflow/correction_loop.py workflow/execution_context.py workflow/orchestration_engine.py workflow/output_validation.py workflow/prima_workflow.py workflow/task_router.py workflow/workflow_state.py
All checks passed.

Focused strict typing:
venv\Scripts\python.exe -m mypy reasoning/reflection_advisor.py reasoning/controller.py reflection/reasoning_reflection_adapter.py reflection/reflection_result.py workflow/correction_loop.py workflow/output_validation.py workflow/execution_context.py workflow/orchestration_engine.py workflow/prima_workflow.py runtime/contracts.py runtime/prima_runtime.py
Success: no issues found in 11 source files. The narrowed run emitted one informational unused-override note for `benchmarks.preflight`.

Final complete CI:
Reported successful by the user on 2026-08-09. Exact output was not captured by Codex, so the focused command results above remain the reproducible recorded evidence.
```

## Risks and deferred work

- Validation rules are deterministic structural checks, not learned factuality or logical-verification claims.
- Reflection advice confidence and utility are not yet calibrated against production outcomes.
- One correction intervention is the conservative default; deployments can inject another validated budget without changing the workflow boundary.
- Graph reasoning, maintenance consumers, procedural memory and benchmark migration remain outside this phase.
- The dirty `benchmarks/locomo-v2/external` submodule is pre-existing and remains untouched.
- `C:\PRIMA_integrated\Draft.png` remains the target architecture, not evidence that deferred diagram components are active.

Stop here. Do not begin the next phase.
