# Phase 08 — Background Memory Pipeline

## Outcome

Phase 08 connects the target diagram's dashed cold path without adding a broker or a second production pipeline. Workflow routes now publish admitted-memory events through a final `MAINTENANCE_ENQUEUE` phase. Publication uses bounded `put_nowait` queueing and never awaits encoding, salience, consolidation, abstraction, graph refresh, retention decay, or forgetting.

`Draft.png` remains the target architecture, not a literal current call graph. This phase closes the maintenance reachability gap; it does not claim that the still-deferred long-term compression block exists.

## Contract and behavior changes

- Added versioned typed event names for admission, encoding, salience, buffering, consolidation, abstraction, graph/index refresh, decay, and terminal failure.
- Added `BackgroundMaintenanceSupervisor` with bounded queue capacity, bounded retries, cooperative cancellation, restart, deterministic flush, event-ID duplicate rejection, and local replaceable dependencies.
- Added production JSONL failure persistence and in-memory test/benchmark failure storage. Failures are also represented as `MAINTENANCE_FAILED` events and exposed by runtime diagnostics.
- Added a real `MaintenancePipeline` over the existing embedding pipeline, `SalienceManager`, `ConsolidationEngine`, retention/forgetting policies, `MemoryEvolutionEngine`, `SemanticAbstractionEngine`, and graph repository.
- Added a real bounded short-term maintenance buffer. No repository label is presented as proof of buffering.
- Added explicit async runtime lifecycle methods: `start_maintenance`, `flush_maintenance`, `stop_maintenance`, and `apply_maintenance_barrier`.
- Added benchmark modes `disabled`, `eventual`, `flush_after_conversation`, `flush_before_question`, and `flush_before_finalization`. The generic runner applies barriers before questions and before finalization, so relevant maintenance cannot continue while final metrics are produced.
- Benchmark maintenance defaults to `disabled` to preserve historical benchmark behavior; selected modes are recorded in manifests/state.
- Added `RuntimeComponent.MAINTENANCE_EVENTS` execution proof plus queue, retry, rejection, duplicate, completion, and failure details in `RuntimeDiagnostics`.

## Ownership and boundaries

- Workflow owns admission-event publication.
- The runtime owns supervisor construction and lifecycle.
- The supervisor owns scheduling, retry, cancellation, idempotency, and failure recording.
- Existing memory engines own their domain operations.
- Benchmark packages select consistency timing only; production packages import no benchmark code.
- Heavy synchronous engines run through `asyncio.to_thread`, outside the request task.

## Acceptance evidence

Focused tests prove queue saturation, duplicate rejection, bounded retries, durable failures, cancellation/restart, cross-event-loop restart, deterministic barriers, pre-finalization benchmark barriers, real engine reachability, and runtime failure visibility.

## Commands and exact results

Focused checks during implementation:

```text
venv\Scripts\python.exe -m compileall -q events memory/maintenance workflow runtime benchmarks/common
EXIT_CODE=0

venv\Scripts\python.exe -m pytest -q tests/memory/test_phase08_background_maintenance.py tests/benchmarks/test_maintenance_barriers.py tests/runtime/test_canonical_contract.py tests/workflow/test_workflow_phase1.py tests/benchmarks/test_hotpotqa.py tests/benchmarks/test_hotpotqa_qol.py
40 passed in 11.13s

venv\Scripts\ruff.exe check events memory/maintenance workflow runtime benchmarks/common benchmarks/locomo/experiment.py benchmarks/hotpotqa/experiment.py tests/memory/test_phase08_background_maintenance.py tests/benchmarks/test_maintenance_barriers.py
All checks passed!

venv\Scripts\python.exe -c "from mypy.main import main; main()" events runtime workflow benchmarks/common memory/maintenance
Success: no issues found in 44 source files
```

Single final quality-gate pass:

```text
venv\Scripts\python.exe -m benchmarks.preflight --json
schema_version=1.0; rouge_l=true; bertscore=true; goemotions_encoder=true; EXIT_CODE=0

venv\Scripts\python.exe -m compileall -q -x '(^|[\\/])(\.git|\.venv|venv|external)([\\/]|$)' .
EXIT_CODE=0

venv\Scripts\python.exe -m pytest -q
........................................................................ [ 27%]
........................................................................ [ 54%]
....................
The desktop command wrapper yielded without its child-session ID and dropped the remaining console text. The process completed; pytest cache recorded 272 collected node IDs and an empty `lastfailed` object. The suite was not rerun, per the one-CI-pass instruction.

venv\Scripts\ruff.exe check .
All checks passed!; EXIT_CODE=0

venv\Scripts\python.exe scripts/run_xenon.py
Production source roots only; EXIT_CODE=0

venv\Scripts\python.exe -m mypy .
Success: no issues found in 216 source files; EXIT_CODE=0

venv\Scripts\python.exe -m bandit -q -c bandit.yaml -r .
One existing `nosec` warning at benchmarks/hotpotqa/experiment.py:67; EXIT_CODE=0
```

## Intentionally deferred risks

- The supervisor is local and replaceable by design; queued work is not durable across a process crash. Production terminal failures are durable.
- Successful event-ID completion history is runtime-lifetime state. Durable replay checkpoints are deferred until a real recovery requirement exists.
- `ConsolidationEngine.run()` scans the repository for each admitted event. Queue bounds protect the hot path, but batching may be needed after production-scale measurement.
- The diagram's dedicated long-term compression block remains target architecture only.

## Changed files

- `README.md`
- `benchmarks/common/{agent.py,runner.py,runtime_adapter.py}`
- `benchmarks/hotpotqa/{checkpoint.py,experiment.py}`
- `benchmarks/locomo/experiment.py`
- `docs/architecture/ADR_CANONICAL_RUNTIME.md`
- `docs/implementation/{STATUS.md,phase_08_background_memory_pipeline.md}`
- `events/{__init__.py,event.py,event_types.py,maintenance_events.py}`
- `memory/maintenance/{__init__.py,background_supervisor.py,maintenance_pipeline.py}`
- `runtime/{contracts.py,prima_runtime.py}`
- `workflow/{execution_context.py,orchestration_engine.py,prima_workflow.py,task_router.py,workflow_state.py}`
- `tests/benchmarks/test_maintenance_barriers.py`
- `tests/memory/test_phase08_background_maintenance.py`
