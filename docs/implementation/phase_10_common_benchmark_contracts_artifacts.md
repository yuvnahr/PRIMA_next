# Phase 10 — Common Benchmark Contracts and Crash-Safe Artifacts

## Outcome

Phase 10 adds benchmark-neutral campaign infrastructure without changing HotpotQA, LoCoMo, or GoEmotions gold data, metrics, prompts, routing, or execution behavior. Individual benchmark migration remains deferred to its owning phases.

`Draft.png` remains the target runtime architecture, not the current call graph. This phase affects benchmark control and artifact ownership only; production packages still import no benchmark code.

## Contracts

The common package now exposes versioned typed contracts for:

- `BenchmarkSpec`, `BenchmarkCase`, `ClassificationCase`, and `ConversationQACase`;
- `BenchmarkMode`, `RunStatus`, and the ordered `LifecycleStage` sequence;
- `PredictionRecord`, `FailureRecord`, `CheckpointRecord`, `BenchmarkManifest`, `BenchmarkSummary`, and `CampaignReference`;
- per-item `ItemTiming` and `TokenUsage`;
- shared `FailureCategory` values plus benchmark-owned `subcode` strings;
- `ProgressEvent`, `ProgressSink`, and forward-only `BenchmarkLifecycle` publication.

Classification retains tuple label predictions and classification gold labels. Conversation QA retains conversation turns, questions, optional answer text, and optional evidence. Neither task is forced into the other's shape.

## Artifact and resume behavior

- `ArtifactLayout` defines fixed `manifest.json`, `summary.json`, and checkpoint/prediction/failure JSONL locations beneath one campaign root.
- `atomic_write_json` writes and flushes a same-directory temporary file before `os.replace`; failed replacement leaves the previous artifact unchanged and removes the temporary file.
- JSONL records are written as one flushed/fsynced line. The store uses a process-local lock and suppresses duplicate case IDs.
- Malformed JSON or JSONL is a hard resume error; corrupt rows are never silently skipped.
- Stable case IDs use a canonical JSON identity hash.
- Manifest fingerprints cover every compatibility field while excluding only campaign identity, mutable status/timestamps, and resume lineage.
- Exact resume rejects source, dataset, selection, provider/model/revision, generation, runtime profile, capability, repository, prompt, seed, dependency, Python, or hardware drift.
- Compatible resume appends a `CampaignReference` to immutable lineage.
- Campaign manifests initialize as non-complete. Complete finalization requires the exact selected terminal case set and matching summary counts. The summary is replaced first and the authoritative manifest completion marker last.
- Failed, cancelled, partial, and complete statuses are explicit.
- Manifest credential fields and credential-bearing URLs are rejected. JSON and JSONL writes redact nested secrets, authorization values, and URL query credentials.

## Scope boundary

This phase intentionally does not replace HotpotQA's existing checkpoint module or alter LoCoMo/GoEmotions runners. The common contracts are ready for all three shapes; adoption belongs to the individual benchmark migration phases so historical behavior does not change prematurely.

## Verification

Focused checks:

```text
venv\Scripts\python.exe -m pytest -q tests/benchmarks/test_common_artifacts.py
12 passed in 0.57s

venv\Scripts\python.exe -m pytest -q tests/benchmarks/test_common_artifacts.py tests/benchmarks/test_hotpotqa.py tests/benchmarks/test_hotpotqa_qol.py tests/benchmarks/test_maintenance_barriers.py
30 passed in 1.87s

venv\Scripts\python.exe -m ruff check benchmarks/common tests/benchmarks/test_common_artifacts.py
All checks passed!

venv\Scripts\python.exe -m mypy benchmarks/common
Success: no issues found in 10 source files
```

Single final quality-gate pass:

```text
venv\Scripts\python.exe -m benchmarks.preflight --json
schema_version=1.0; rouge_l=true; bertscore=true; goemotions_encoder=true; EXIT_CODE=0

venv\Scripts\python.exe -m compileall -q -x '(^|[\\/])(\.git|\.venv|venv|external)([\\/]|$)' .
EXIT_CODE=0

venv\Scripts\python.exe -m pytest -q
281 passed, 1 skipped in 131.03s; EXIT_CODE=0

venv\Scripts\python.exe -m ruff check .
All checks passed!; EXIT_CODE=0

venv\Scripts\python.exe scripts/run_xenon.py
Production source roots only; EXIT_CODE=0

venv\Scripts\python.exe -m mypy .
Success: no issues found in 221 source files; EXIT_CODE=0

venv\Scripts\python.exe -m bandit -q -c bandit.yaml -r .
One existing nosec warning at benchmarks/hotpotqa/experiment.py:67; EXIT_CODE=0
```

## Intentionally deferred risks

- Duplicate suppression is process-local. Add an OS file lock only if a later benchmark phase adopts multiple writer processes; existing benchmark parallelism is thread-based.
- Atomicity is per file. `manifest.json` is the authoritative completion marker and is written last; readers must not infer completion from a summary file alone.
- Existing benchmark-specific artifact layouts remain until their migration phases.

## Changed files

- `benchmarks/common/{__init__.py,artifacts.py,contracts.py,lifecycle.py}`
- `docs/architecture/ADR_CANONICAL_RUNTIME.md`
- `docs/implementation/{STATUS.md,phase_10_common_benchmark_contracts_artifacts.md}`
- `pyproject.toml`
- `tests/benchmarks/test_common_artifacts.py`

The pre-existing `benchmarks/locomo-v2/external` submodule marker remains untouched. Test-regenerated historical evaluation JSON was restored to `HEAD` and is not part of this phase.
