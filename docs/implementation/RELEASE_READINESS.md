# Release Readiness

Date: 2026-08-11

Runtime: Python 3.10.11 from `venv\Scripts\python.exe`

Scope: Phase 15 validation; no full-dataset GPU execution

## Decision

**GO for the validated code and fixture-campaign release boundary. NO-GO for a
full-dataset GPU campaign until the deployment-specific configuration, endpoint,
datasets, and hardware are supplied and pass preflight.**

The repository-wide quality gates and complete fixture campaign are green in the
final Phase 15 worktree.

## Command transcript

Focused Phase 15 validation:

```text
venv\Scripts\python.exe -m pytest tests/integration/test_release_readiness.py -q
5 passed in 8.45s

venv\Scripts\python.exe -m pytest tests/benchmarks/test_campaign.py -q -k sigterm
1 passed, 7 deselected in 17.59s

venv\Scripts\python.exe -m pytest tests/benchmarks/test_campaign.py -q -k sigint
1 passed, 7 deselected in 17.73s

venv\Scripts\python.exe -m ruff check benchmarks/campaign tests/benchmarks/test_campaign.py tests/integration/test_release_readiness.py
All checks passed!

venv\Scripts\python.exe -m mypy benchmarks/campaign
Success: no issues found in 12 source files
```

Complete checked-in smoke campaign:

```text
venv\Scripts\python.exe -m benchmarks.campaign.cli run --config benchmarks/campaign/smoke.yaml
campaign_id: campaign-9a2aa671-7133-419e-83b2-b667d5db5c86
status: complete
modes: goemotions-model-only complete; hotpotqa-model-only complete; locomo-model-only complete
execution failures: 0
request attempts/successes/retries/timeouts: 4/4/0/0
maximum active inference requests: 1
```

Artifact audit of 37 smoke files:

```text
absolute developer-path matches: 0
credential-pattern matches: 0
GoEmotions checkpoint/prediction rows: 1/1, unique IDs: 1/1
HotpotQA checkpoint/prediction rows: 2/2, unique IDs: 2/2
LoCoMo checkpoint/prediction rows: 1/1, unique IDs: 1/1
```

The full repository quality-gate transcript is recorded after the single final CI
pass:

```text
venv\Scripts\python.exe -m pip check
No broken requirements found.

venv\Scripts\python.exe -m benchmarks.preflight --json
rouge_l available; bertscore available; goemotions_encoder available

venv\Scripts\python.exe -m compileall -q -x '(^|[\\/])(\.git|\.venv|venv|external)([\\/]|$)' .
passed (no output)

venv\Scripts\python.exe -m pytest -q
315 passed, 1 skipped in 206.37s (0:03:26)

venv\Scripts\python.exe -m ruff check .
All checks passed!

venv\Scripts\python.exe scripts/run_xenon.py
passed on the production source-root allowlist

venv\Scripts\python.exe -m mypy .
Success: no issues found in 233 source files

venv\Scripts\python.exe -m bandit -q -c bandit.yaml -r .
passed; two existing valid nosec annotations produced informational warnings
```

## Fault-injection evidence

| Fault | Evidence and expected result |
|---|---|
| Provider timeout | Phase 15 integration test proves two attempts for one retry and timeout telemetry of two. |
| Rate limit | Denying limiter raises before provider execution; no fallback prediction is emitted. |
| Malformed/truncated structured output | Both payloads yield typed failed outcomes with `fallback_used=false`. |
| Retrieval failure | Controller boundary test returns a structured retrieval failure distinct from no evidence. |
| Graph capability missing | Retrieval alignment test reports the unavailable capability explicitly. |
| Memory-store failure | Ingestion returns a typed failed outcome and does not generate. |
| Event-worker failure | Background-maintenance tests record typed failure, retry, and restart behavior. |
| Disk-write interruption | Common artifact tests preserve the previous file and partial manifest. |
| Duplicate checkpoint | Artifact store rejects duplicate terminal case IDs. |
| Incompatible resume | Manifest fingerprint/schema mismatch is rejected. |
| SIGINT/SIGTERM | Real subprocess tests checkpoint one item, terminate, resume, and finish with unique records. |
| GPU semaphore bound | Concurrent requests observe a maximum of one active inference request. |

## Architecture truth matrix

| Route or component | Verified behavior |
|---|---|
| Full conversation profile | Parser, state load, affect, dense/graph retrieval, planning, symbolic world simulation, uncertainty, triggered reflection, generation/action, validation, state/memory commit, and maintenance enqueue execute in one trace. |
| Reflection | A low-confidence injection triggers reflection; the output remains generated rather than echoed. Existing correction tests prove accepted advice can revise retrieval/execution. |
| Diagnostics | Executed/skipped components, trace count, and requested/active reranker identities reconcile. |
| Document ingestion | Stores through the canonical route and skips model generation. |
| GoEmotions | Uses `EMOTION_CLASSIFICATION`/`affect_only`; QA retrieval, world simulation, planning, and tools are excluded. |
| `model_only` | Skips retrieval components. |
| Benchmark adapters | All three submit typed requests through `PrimaRuntime.execute()`; production imports no benchmark packages. |

## Benchmark readiness

| Benchmark | Resume | Core status | Claim boundary |
|---|---|---|---|
| GoEmotions | Per example | Fixture campaign complete | Affect/classification component benchmark only; parse recovery and trained baselines are separate. |
| HotpotQA | Per case | Fixture campaign complete | Supplied distractor context is not open-domain retrieval; scoring failures are distinct from execution failures. |
| LoCoMo | Per question | Fixture campaign complete; optional semantic metrics disabled | Preview is not a full run; core metrics require no ROUGE/BERTScore package. |

Gold data remains evaluator-owned in scoring/checkpoint artifacts. Source and request
boundary tests prove expected answers, labels, supporting facts, and evidence IDs do
not enter prompts, retrieval, routing, or reflection.

## Required no-go audit

No required no-go condition is present in the validated scope: output is generated,
QA uses the workflow, reflection can alter execution, diagnostics reconcile, all
benchmarks resume, LoCoMo core metrics are dependency-independent, invalid pairs are
refused, inference is bounded, interruption survives resume, and no P0 finding is
open. The production campaign remains deliberately blocked until real environment
fields pass preflight.

## Unresolved risks

- The fake provider proves orchestration and reliability, not answer quality.
- No real endpoint, model revision, full split, or GPU was exercised in Phase 15.
- Host telemetry reports unavailable when optional `psutil` is absent; this is
  non-fatal and explicit.
- Windows SIGINT-equivalent testing uses the supported console-break event because a
  detached Python process does not reliably receive terminal Ctrl+C injection.
- Real-environment model load, GPU peak memory, endpoint rate limits, and throughput
  must be re-measured in the deployment campaign.
