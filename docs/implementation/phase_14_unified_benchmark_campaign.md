# Phase 14 — Unified Three-Benchmark Campaign

## Outcome

One command now runs configured GoEmotions, HotpotQA, and LoCoMo modes as one campaign:

```powershell
python -m benchmarks.campaign.cli run --config <path> --resume
```

The campaign owns one root run ID, one provider/client session, and one inference semaphore. Benchmark workers still receive isolated canonical `PrimaRuntime` instances, but those runtimes are injected with the same client and never construct independent model providers. The default inference bound is one active request.

## Configuration and preflight

`CampaignConfig` is a frozen, extra-forbidding Pydantic contract loaded from YAML or JSON. It validates benchmark/profile/variant routes, unique IDs, scheduler concurrency authority, and statically knowable paired invariants. Exact `${NAME}` placeholders resolve only from the supplied process environment; an unresolved value fails before execution.

Preflight verifies dataset presence and SHA-256 hashes, all GoEmotions split files, provider endpoint health, context budgets, structured-output and seed capability, optional metric availability, output writes, free disk, repository mode, and the GPU/API scheduling policy. `full_gpu.example.yaml` intentionally cannot run until deployment-specific model, revision, endpoint, data, and output fields are supplied.

## Scheduling, isolation, and resume

Sequential scheduling is the default. Interleaved mode overlaps CPU-side loading, serialization, and scoring through a shared queue while all inference still passes through the campaign semaphore. More than one active request requires either enough explicitly listed GPU devices or explicit API-concurrency validation.

Each benchmark/mode writes beneath `modes/<mode-id>` using its existing per-item artifact store. The root manifest records pending, running, complete, partial, or failed state independently for every mode. Resume requires the identical normalized config, retains the root campaign ID, skips complete modes, and invokes child resume only when that child's manifest exists. Existing child stores suppress duplicate case IDs. Ordinary item failures remain terminal benchmark records; an isolated mode exception is recorded without corrupting siblings when policy permits continuation.

## Comparisons, aggregation, and telemetry

Paired comparisons require the same benchmark, model/revision, selected IDs and order, seed, generation configuration, context budget, and retry policy. A mismatch is reported as `refused`; it is never emitted as a valid delta. Valid pairs include the configured metric delta, corrected/regressed/unchanged case counts, and a deterministic paired-bootstrap 95% interval.

Only complete or partial validated modes enter `aggregate.json`. The campaign also writes `report.md` and per-mode summaries. Telemetry includes provider-session initialization time, request attempts, retries/timeouts, p50/p95/p99 latency, prompt/completion tokens, throughput, maximum active inference, CPU/RAM snapshots, and optional GPU utilization plus peak-observed memory. Telemetry imports and device probes are best effort and cannot decide campaign success.

## Checked-in configurations and coverage

- `smoke.yaml` runs fixture subsets of all three benchmarks through their canonical public runtime with the fake provider.
- `full_gpu.example.yaml` is a non-runnable deployment template with unresolved environment placeholders.
- Focused tests exercise the three-benchmark smoke campaign, resume/duplicate suppression, inference bounds, isolated mode failure, and invalid comparison configuration.

## Verification

Focused validation completed with 6 passing campaign tests plus focused Ruff and mypy checks. The one final repository-wide CI pass used Python 3.10.11: benchmark capability preflight passed, owned-source compileall passed, 308 tests passed with 1 optional skip, Ruff passed, mypy passed across 233 source files, Bandit passed after an explicit deterministic-bootstrap exemption, and production-only Xenon passed.

## Risks and deferrals

- The fake-provider smoke campaign validates orchestration and runtime routing, not model quality.
- Remote endpoints cannot expose a portable true model-load duration; `model_load_time_ms` is explicitly labeled as shared provider-session initialization.
- GPU peak memory is the maximum observed campaign boundary snapshot, not a high-frequency profiler trace.
- No live model, full benchmark dataset, or multi-GPU campaign was run during this implementation phase.
