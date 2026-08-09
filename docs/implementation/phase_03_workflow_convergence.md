# Phase 03 — Workflow Convergence

## Outcome

Conversation, factual QA and document ingestion now enter through `PrimaRuntime.execute(PrimaRequest)` and run one workflow-owned task/profile lifecycle. The compatibility methods `process`, `process_async`, `answer_question` and `ingest_document` only construct requests, delegate to `execute`, and convert the typed response back to `RuntimeResult`, `AnswerResult` or `MemoryNote`.

## Behavior and contract changes

- Added `ExecutionOutcome`: `answered`, `abstained`, `failed`, `ingested`, `classified`, `cancelled`.
- Added workflow phases for evidence acquisition, answer generation, document ingestion and memory commit.
- Moved bounded `ReasoningController` use into `EvidenceAcquisitionController`; runtime QA no longer calls it directly.
- Added `AnswerGenerationController`, which consumes the request, selected evidence, plan and cognitive state and calls one injected `LLMClient`.
- Provider/validation failures return a typed failed result; factual insufficiency returns a typed abstention with human-readable text, never an encoded JSON answer.
- Restricted `OutputController` to shaping an existing typed result. Missing results fail explicitly and there is no input echo fallback.
- Ingestion validates, embeds and indexes semantic memory inside the workflow and never selects generation under `ingestion_only`.
- Conversation memory admission/commit now occurs as a workflow phase.
- Workflow failures carry their failed context through `WorkflowExecutionError`; cancellation remains a distinct typed terminal outcome.

## Route behavior

All nine valid Phase 02 task/profile combinations have deterministic workflow phase plans. Model-only conversation/QA routes generate directly; simple-RAG routes acquire evidence then generate; full routes also select affect, planning, reflection and action. World, uncertainty, graph and maintenance-event behavior remains explicitly deferred and skipped.

## Long-horizon smoke correction

`test_long_horizon_runner_smoke` became slow because `_simulate_user` hard-coded `PrimaRuntime()`. Real generation then caused unavailable-provider retries for every one of 60–80 turns. `LongHorizonRunner` now accepts a runtime factory, and non-provider smoke tests inject a deterministic client. The isolated smoke completes in about 11 seconds and the full suite reports it at about 10 seconds instead of stalling on provider retries.

## Files changed

- `runtime/contracts.py`, `runtime/__init__.py`, `runtime/prima_runtime.py`
- `workflow/answer_generation.py`, `workflow/prima_workflow.py`, `workflow/task_router.py`
- `workflow/execution_context.py`, `workflow/workflow_state.py`, `workflow/orchestration_engine.py`, `workflow/__init__.py`
- `evaluation/runners/long_horizon_runner.py`
- Focused runtime, workflow, reasoning, evaluation and integration tests
- `docs/architecture/ADR_CANONICAL_RUNTIME.md`
- `docs/implementation/ARCHITECTURE_TRUTH_MATRIX.md`
- `docs/implementation/STATUS.md`
- `docs/implementation/phase_03_workflow_convergence.md`

## Commands and results

```text
venv\Scripts\python.exe -m pytest -q tests\integration\test_runtime_pipeline.py
Initial characterization: collection failed because ExecutionOutcome did not exist.

venv\Scripts\python.exe -m pytest -q tests\integration\test_runtime_pipeline.py tests\runtime\test_canonical_contract.py tests\workflow\test_workflow_phase1.py tests\reasoning\test_runtime_integration.py
17 passed in 4.66s after workflow convergence and legacy expectation updates.

venv\Scripts\python.exe -m pytest -q tests\evaluation\test_long_horizon.py::test_long_horizon_runner_smoke
1 passed; measured command duration 11.07s.

venv\Scripts\python.exe -m pytest -q tests\evaluation\test_long_horizon.py::test_long_horizon_runner_output_files
1 passed in 8.62s.

venv\Scripts\python.exe -m pytest -q --durations=10
225 passed in 105.24s before the final classified/cancelled and route-plan assertions were added.

venv\Scripts\python.exe -m pytest -q tests\integration\test_runtime_pipeline.py tests\runtime\test_canonical_contract.py tests\reasoning\test_runtime_integration.py
14 passed in 4.46s after typed failure/cancellation propagation.

venv\Scripts\python.exe -m ruff check runtime workflow ...
All checks passed.

venv\Scripts\python.exe -m mypy runtime workflow
Success: no issues found in 17 source files.
```

## Final quality gates

```text
venv\Scripts\python.exe -m pytest -q tests\integration\test_runtime_pipeline.py tests\runtime\test_canonical_contract.py tests\reasoning\test_runtime_integration.py
16 passed in 4.99s.

venv\Scripts\python.exe -m pytest -q --durations=10
228 passed in 59.54s; long-horizon smoke 6.14s.

C:\Users\Yuv Nahar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m compileall -q -x '(^|[\\/])(\.git|\.venv|venv|external)([\\/]|$)' .
Exit 0; no output; 0.4s.

venv\Scripts\python.exe -m ruff check .
All checks passed.

venv\Scripts\python.exe -m mypy .
Success: no issues found in 205 source files.

venv\Scripts\python.exe -m bandit -c bandit.yaml -r .
No issues identified; 23,712 lines scanned; 14 disabled checks; 0 files skipped.

venv\Scripts\python.exe scripts\run_xenon.py
Exit 0 in 1.3s; production PRIMA roots only.
```

## Risks and deferred work

- Full-profile route plans still select target components—world simulation, uncertainty, graph and maintenance—that Phase 03 does not implement; diagnostics report them skipped.
- Reflection remains a single checkpoint and does not yet retrieve/replan/correct; Phase 04 owns that loop.
- GoEmotions remains benchmark-local until the benchmark adapter phase.
- Legacy private answer-formatting helpers remain for existing low-level compatibility tests but have no production call site; all public QA orchestration is workflow-owned.
- The complete test environment remains Python 3.10.11; compileall uses the bundled Python 3.12 runtime.
- Stop here. Do not begin Phase 04.
