# Phase 04 — Cognitive State and Memory Ownership

## Outcome

Canonical requests now use workflow-owned state load/commit phases, typed versioned cognitive state, explicit runtime modes, and mode-enforced memory repository selection. Conversation and factual QA share state by session ID. Production fails before execution unless persistent ChromaDB memory and persistent cognitive-state storage are configured.

## Contract and behavior changes

- `CognitiveState` contains typed goal, emotional, task, confidence and environment sections, metadata, version, and UTC timestamps. Legacy `*_state` access remains a compatibility alias.
- `StateManager` defines `load`, `apply_delta`, `save`, and `reset`; deterministic in-memory and versioned JSON implementations enforce optimistic expected versions.
- `RuntimeMode` is `test`, `benchmark`, or `production`. Diagnostics and `runtime_manifest()` expose mode, memory backend, persistence and state version.
- The repository factory permits ephemeral memory only outside production, requires benchmark declaration, and makes valid ChromaDB the production memory source of truth.
- All canonical routes load state. Task-appropriate successful and abstained routes commit it; failed/cancelled routes explicitly do not.
- Ingestion records semantic-source admission. Admitted successful conversations create paired user/assistant episodic records. Failed, cancelled, abstained, QA and classification no-write decisions are explicit.
- The existing opt-in Ollama smoke preflight now verifies the configured model is installed instead of treating any listener on the port as usable.
- Xenon remains restricted to production source roots and terminates after 30 seconds; CI also caps the step at one minute.

## Files changed

- State: `state/cognitive_state.py`, `state/state_manager.py`, `state/__init__.py`
- Runtime/storage policy: `config/runtime_mode.py`, `memory/repository_factory.py`, `runtime/contracts.py`, `runtime/prima_runtime.py`, `runtime/__init__.py`
- Workflow: `workflow/workflow_state.py`, `workflow/task_router.py`, `workflow/orchestration_engine.py`, `workflow/prima_workflow.py`
- Benchmark declaration: `benchmarks/common/runtime_adapter.py`, `benchmarks/hotpotqa/experiment.py`, `benchmarks/locomo/experiment.py`
- Quality-gate bound: `scripts/run_xenon.py`, `.github/workflows/ci.yml`
- Focused state, workflow, runtime, benchmark and production-smoke tests
- Architecture ADR, truth matrix and implementation status ledger

## Verification

```text
Phase 03 baseline focused suite:
16 passed in 34.84s.

Focused Phase 04/regression suite:
42 passed in 12.78s.

Initial full suite:
232 passed, 1 failed in 130.33s. A live Ollama server lacked the requested default `llama3.1` model and returned HTTP 404. No fallback was added.

venv\Scripts\python.exe -m pytest -q --durations=10
233 passed, 1 skipped in 103.98s. The pre-existing opt-in smoke reported the missing endpoint/model capability.

Python 3.12: python.exe -m compileall -q -x '(^|[\\/])(\.git|\.venv|venv|external)([\\/]|$)' .
Exit 0; no output. The first sandboxed attempt could not replace bytecode caches; the approved workspace run passed.

venv\Scripts\python.exe -m ruff check .
All checks passed.

venv\Scripts\python.exe -m mypy .
Success: no issues found in 208 source files.

venv\Scripts\python.exe -m bandit -c bandit.yaml -r .
No issues identified; 24,193 lines scanned; 14 disabled checks; 0 files skipped.

venv\Scripts\python.exe scripts\run_xenon.py
Exit 0 in seconds; production source roots only with a 30-second process timeout and one-minute CI timeout.

venv\Scripts\python.exe -m pytest -q tests\scripts\test_run_xenon.py
2 passed in 0.15s, including forced-timeout exit 124 behavior.
```

## Risks and deferred work

- JSON state writes use a process-local lock; multi-process writers require a transactional state adapter before horizontal scaling.
- Document insertion and cognitive-state commit are not one cross-store transaction; reconciliation belongs with later maintenance/event work.
- Procedural memory, graph activation, maintenance events, world simulation, uncertainty gating and bounded reflection remain disconnected.
- The full test environment remains Python 3.10.11; compileall and declared targets use Python 3.12.
- `C:\PRIMA_integrated\Draft.png` remains the target architecture, not a current-call-graph claim.

Stop here. Do not begin the next phase.
