# Phase 05 — Planning, Symbolic World Simulation and Uncertainty Gate

## Outcome

Implementation and verification are complete. The canonical `prima_full` workflow now routes planning output through the existing deterministic symbolic world model, aggregates available confidence signals, and applies one configurable typed execution gate before reflection, retrieval retry, clarification, abstention, or action/model execution. Shorter profiles remain on the same public runtime boundary and explicitly diagnose why world simulation and uncertainty are disabled.

## Contract and behavior changes

- Added workflow phases `WORLD_SIMULATION`, `UNCERTAINTY_ESTIMATION`, and `EXECUTION_DECISION` after planning for full conversation, factual-QA, and tool routes.
- Added typed gate decisions: `continue`, `reflect`, `retry_retrieval`, `ask_for_clarification`, and `abstain`.
- Added injected, validated `UncertaintyGatePolicy` thresholds. Every gate result and runtime diagnostic records the exact thresholds and rule used.
- World simulation uses only `StateSimulator` and its deterministic symbolic rules. Inputs include typed cognitive state, selected plan, execution policy constraints, and intended LLM/tool target. No learned-prediction claim is made.
- Uncertainty uses available retrieval, planner, affect, state, policy, and prior-reflection signals. Missing state/reflection observations are omitted rather than fabricated.
- The orchestration engine owns the branch. Retrieval retry re-enters evidence acquisition, planning, simulation, uncertainty and decision once by default; the configured limit is independently enforced by the workflow.
- Reflection executes only for a `reflect` decision. Clarification and high-risk denial skip action and generation and produce a typed abstention.
- `ActionController` receives typed world prediction and uncertainty fields directly from `ExecutionContext`; it no longer reads unset metadata placeholders.
- `RuntimeDiagnostics` now reports the final decision, rule, thresholds, bounded decision history, and enabled/disabled details for world and uncertainty components.
- `SimulationContext` now deduplicates unhashable typed plan constraints safely.
- The project contract, Ruff, mypy and GitHub CI now consistently target Python 3.10.

## Changed files

- Gate contract and exports: `uncertainty/execution_gate.py`, `uncertainty/__init__.py`
- Workflow state and orchestration: `workflow/workflow_state.py`, `workflow/execution_context.py`, `workflow/task_router.py`, `workflow/orchestration_engine.py`, `workflow/prima_workflow.py`
- Runtime contract, injection and diagnostics: `runtime/contracts.py`, `runtime/prima_runtime.py`
- Symbolic simulation input handling: `world/simulation_context.py`
- Focused branch tests: `tests/workflow/test_phase05_confidence_branch.py`
- Architecture ledger: `docs/implementation/ARCHITECTURE_TRUTH_MATRIX.md`, `docs/implementation/STATUS.md`
- Python target configuration: `codex_rules.txt`, `pyproject.toml`, `.github/workflows/ci.yml`

## Verification

```text
Focused regression selection:
venv\Scripts\python.exe -m pytest -q tests/workflow/test_phase05_confidence_branch.py tests/runtime/test_canonical_contract.py tests/integration/test_runtime_pipeline.py
Initial result: 1 failed, 20 passed in 28.38s. The capturing test double used zero-argument `super()` with a slotted dataclass; production behavior was not at fault.

Focused failed-test rerun after correcting the test double:
venv\Scripts\python.exe -m pytest -q tests/workflow/test_phase05_confidence_branch.py::test_high_confidence_executes_with_world_and_uncertainty_inputs
1 passed in 8.35s.

Focused Phase 05 suite after bounded-retry and diagnostic-history coverage:
venv\Scripts\python.exe -m pytest -q tests/workflow/test_phase05_confidence_branch.py
7 passed in 8.60s.

Single final repository-wide run (not repeated, per phase instruction):
C:\Users\Yuv Nahar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m compileall -q -x '(^|[\\/])(\.git|\.venv|venv|external)([\\/]|$)' .
Exit 1. Python 3.12 could not replace eight existing `__pycache__` files: `PermissionError: [Errno 13] Permission denied`.

venv\Scripts\python.exe -m pytest -q
Exit 101 before test collection: the venv launcher could not create the configured Python 3.10 process.

venv\Scripts\python.exe -m ruff check .
Exit 101 before execution for the same launcher failure.

venv\Scripts\python.exe -m mypy .
Exit 101 before execution for the same launcher failure.

venv\Scripts\python.exe -m bandit -c bandit.yaml -r .
Exit 101 before execution for the same launcher failure.

venv\Scripts\python.exe scripts\run_xenon.py
Exit 101 before execution for the same launcher failure; Xenon itself did not start.

Environment evidence after the failed run:
`venv\pyvenv.cfg` still targets `C:\Users\Yuv Nahar\AppData\Local\Programs\Python\Python310`; the configured base executable exists, but the composite sandboxed run could not launch it. Existing Python 3.12 cache files are owned by the current user, so the write denial is execution-environment related rather than a source error.

Requested Python 3.10 CI run:
venv\Scripts\python.exe -m compileall -q -x "(^|[\\/])(\.git|\.venv|venv|external)([\\/]|$)" .
Exit 0; no output.

venv\Scripts\python.exe -m pytest -q
238 passed, 1 skipped, 2 failed in 133.34s. Both failures were legacy routes skipping reflection when no uncertainty decision existed.

Focused regression rerun after fixing the shared skip condition:
venv\Scripts\python.exe -m pytest -q tests/workflow/test_workflow_phase1.py::WorkflowPhase1Test::test_prima_workflow_routes_real_subsystem_outputs_through_context tests/workflow/test_workflow_phase1.py::WorkflowPhase1Test::test_workflow_owns_lifecycle_routes_outputs_and_events
2 passed in 7.84s.

venv\Scripts\python.exe -m ruff check .
Exit 1: two import-order findings in Phase 05 files; both were corrected.

Focused Ruff verification after correction:
venv\Scripts\python.exe -m ruff check runtime/prima_runtime.py workflow/prima_workflow.py workflow/orchestration_engine.py uncertainty/execution_gate.py tests/workflow/test_phase05_confidence_branch.py
All checks passed.

venv\Scripts\python.exe -m mypy .
Exit 1 after checking 209 files: two `union-attr` findings in `OutputController`; both were corrected.

Focused mypy verification after correction:
venv\Scripts\python.exe -m mypy workflow/prima_workflow.py workflow/orchestration_engine.py runtime/prima_runtime.py uncertainty/execution_gate.py
Success: no issues found in 4 source files. One informational note reported an unused override for `benchmarks.preflight` in this narrowed invocation.

venv\Scripts\python.exe -m bandit -q -c bandit.yaml -r .
Exit 0; only the configured `nosec` warning for `benchmarks/hotpotqa/experiment.py:66` was emitted.

venv\Scripts\python.exe scripts\run_xenon.py
Exit 0; production source roots only.

Final complete verification:
The user manually reran CI after the focused fixes and confirmed that every gate passed. Exact output was not captured in this agent session and is therefore not reproduced here.
```

## Risks and deferred work

- Gate thresholds are explicit and deterministic but are not yet calibrated against production observations.
- Reflection is now conditionally reached, but reflection-driven revised-query advice, correction, and post-reflection re-estimation remain under open findings `REFL-001` and `REFL-002`.
- Graph reasoning, maintenance consumers, procedural memory, and semantic reranker attribution remain out of Phase 05 scope and open in the ledger.
- The default retrieval retry limit is one. Deployments may inject another validated policy, but the workflow still enforces the declared bound.
- The project, Ruff, mypy and CI targets were aligned to the available Python 3.10 environment after the initial verification blocker.
- The final complete CI pass was manually verified by the user; its exact command output is not available in this agent session.
- `C:\PRIMA_integrated\Draft.png` remains the target architecture, not a claim that every diagram block is currently active.

Stop here. Do not begin the next phase.
