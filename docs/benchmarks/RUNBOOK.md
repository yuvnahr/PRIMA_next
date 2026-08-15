# Benchmark Campaign Runbook

## Scope

The campaign command runs GoEmotions, HotpotQA, and LoCoMo through one shared
provider session. It does not imply that every task invokes every PRIMA component.
GoEmotions uses the bounded affect route; ingestion never generates; `model_only`
never retrieves.

Use the repository Python 3.10 environment directly:

```powershell
venv\Scripts\python.exe --version
venv\Scripts\python.exe -m benchmarks.campaign.cli run --config benchmarks/campaign/smoke.yaml
venv\Scripts\python.exe -m benchmarks.campaign.cli run --config benchmarks/campaign/smoke.yaml --resume
```

The smoke configuration uses fixture data and a fake provider. Remove or choose a
new output directory before starting a fresh smoke; use `--resume` only for the same
configuration and dataset hashes.

## Production preflight

Copy `benchmarks/campaign/full_gpu.example.yaml` and supply every `${...}` field in
the deployment environment. Preflight rejects unresolved fields, missing or changed
datasets, unavailable endpoints, insufficient context windows, unsupported
structured output, unavailable optional metrics, unwritable output, insufficient
disk, invalid repository modes, and unvalidated concurrency.

Keep `max_gpu_requests: 1` unless the listed GPUs or API concurrency have been
explicitly validated. Never construct providers inside benchmark workers.

## Interruption and resume

SIGINT, SIGTERM, and Windows console-break cancellation return exit code 130 after
the active mode is marked partial or failed. Re-run the identical command with
`--resume`. Resume retains the campaign ID, verifies the manifest fingerprint, and
suppresses case IDs already present in the append-only checkpoint.

Before accepting results, confirm:

- root and child manifests are complete or explicitly partial;
- checkpoint and prediction case IDs are unique;
- processed/completed/failed/category totals reconcile under each benchmark's schema;
- paired comparisons are `valid`, never `refused`;
- telemetry reports `max_active_requests` at or below the configured bound;
- the artifact tree contains no credentials or developer-machine paths.

Partial campaigns may support diagnostics but are not headline result sets.
