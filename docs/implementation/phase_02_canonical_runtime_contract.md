# Phase 02 — Canonical Runtime Contract and Route Profiles

## Outcome

Phase 02 adds the typed canonical boundary without claiming that all legacy execution has migrated.

```text
await PrimaRuntime.execute(PrimaRequest) -> PrimaResponse
```

The contract supports all five task kinds and profiles, versioned JSON serialization, deterministic route selection for all 25 combinations, explicit invalid-route results, component capability data, state deltas, evidence references, and planned/executed/skipped diagnostics.

`PrimaRuntime.execute_sync` uses `asyncio.run` only when no event loop is active. Inside an active loop it raises a clear error directing callers to `await PrimaRuntime.execute(request)`.

## Current execution behavior

| Route | Phase 02 behavior |
|---|---|
| `document_ingestion/ingestion_only` | Completes through the canonical boundary, commits memory, returns no answer text. |
| `emotion_classification/affect_only` | Completes through the canonical boundary using the injected affect engine; no QA components run. |
| Valid conversation, factual-QA and tool routes | Return typed `not_implemented` while recording the selected plan and only proven calls. |
| Invalid task/profile combination | Returns typed `rejected` with every component skipped and a clear error. |

Existing `process`, `process_async`, `answer_question`, and `ingest_document` callers remain operational. Benchmarks and metrics were not changed, and production runtime imports no benchmark package.

## Route validity

Nine combinations are valid: three profiles each for conversation and factual QA, ingestion-only for document ingestion, affect-only for emotion classification, and full PRIMA for tool requests. The other sixteen combinations are deterministic invalid configurations. The complete matrix and component ownership are documented in `docs/architecture/ADR_CANONICAL_RUNTIME.md`.

## Diagnostics contract

- `planned_components`: selected route intent.
- `executed_components`: components actually called by the canonical boundary.
- `skipped_components`: every known component not actually called, including planned but unavailable/deferred components.
- `capabilities`: current canonical availability and a reason when unavailable.

A selected component is never reported as executed merely because a class or route entry exists.

## Files changed

- `runtime/contracts.py`
- `runtime/route_profiles.py`
- `runtime/prima_runtime.py`
- `runtime/__init__.py`
- `tests/runtime/test_canonical_contract.py`
- `tests/scripts/test_run_xenon.py`
- `scripts/run_xenon.py`
- `codex_rules.txt`
- `docs/architecture/ADR_CANONICAL_RUNTIME.md`
- `docs/implementation/ARCHITECTURE_TRUTH_MATRIX.md`
- `docs/implementation/STATUS.md`
- `docs/implementation/phase_02_canonical_runtime_contract.md`

## Commands and exact results

```text
venv\Scripts\python.exe -m pytest -q tests\runtime\test_canonical_contract.py
Initial characterization: collection failed with ModuleNotFoundError: runtime.contracts.
After implementation: 7 passed in 6.88s.

venv\Scripts\python.exe -m pytest -q tests\runtime\test_canonical_contract.py tests\scripts\test_run_xenon.py
Final focused run: 8 passed in 6.65s.

C:\Users\Yuv Nahar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m compileall -q -x '(^|[\\/])(\.git|\.venv|venv|external)([\\/]|$)' .
Exit 0; no output; 0.3 seconds.

venv\Scripts\python.exe -m pytest -q
223 passed in 50.45s.

venv\Scripts\python.exe -m ruff check .
All checks passed.

venv\Scripts\python.exe -m mypy .
Success: no issues found in 204 source files.

venv\Scripts\python.exe -m bandit -c bandit.yaml -r .
No issues identified; 23,133 lines scanned; 14 disabled checks; 0 files skipped.

venv\Scripts\python.exe -m pytest -q tests\scripts\test_run_xenon.py
1 passed in 0.12s.

venv\Scripts\python.exe scripts\run_xenon.py
Exit 0 in 1.2 seconds; analyzed only `main.py` and the allowlisted production packages.
```

An accidental raw repository-root Xenon invocation failed to complete within the 180-second command cap because it descended into non-production trees. Its orphaned processes were stopped, `benchmarks` and `evaluation` were removed from the runner allowlist, `reasoning` was added, and the regression test plus `codex_rules.txt` now require the bounded runner. CI already invokes that runner.

## Risks and deferred work

- The ADR ownership table is the target architecture, not proof of current connectivity.
- Conversation, factual-QA and tool execution are not yet migrated to the canonical workflow; they return `not_implemented` through `execute` while legacy methods remain available.
- The ingestion and affect-only short routes currently invoke established subsystem methods directly. Moving those calls under one workflow-owned lifecycle remains `ARCH-004`/`ARCH-001` closure work.
- World-model, uncertainty, bounded reflection/correction, graph activation and maintenance scheduling were not implemented.
- Full pytest ran on the existing Python 3.10.11 venv; Python 3.12.13 compiled all owned sources. CI remains the authoritative full Python 3.12 run.
- Stop here. Do not begin internal workflow migration in this phase.
