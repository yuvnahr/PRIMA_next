# Kaggle LoCoMo session budget

The canonical `PRIMA_Kaggle_Benchmark.ipynb` preserves all 1,986 questions,
`prima_full`, `normal_prima_admission`, full historical replay,
`flush_before_question`, seed 13, 8192 context tokens, 512 output tokens and
the configured retry policy. The local LoCoMo dataset has ten conversations,
5,882 historical turns and SHA-256
`553cd5a15e25f2ceccc6ed185221eba645080c93e5b91087560a91aa5961f365`.

The supplied reference run reached 698/1,986 questions after 27,049 seconds
with 49,912 seconds remaining: roughly 21h23m projected, before setup and
final artifacts. Its fixed-path, resumable canary is not reliable fresh
throughput evidence. The reference notebook was on another machine; these
are supplied observations, not a reproduced GPU run. The displayed 19.47
GiB was disk space, not a measured RAM peak.

## Running the notebook

The notebook now defaults to an isolated two-example smoke mode. See
[three-benchmark smoke validation](kaggle-smoke-validation.md) for its exact
steps and limitations. The production timing steps below require
`SMOKE_MODE=False` in a new session.

1. Publish the accompanying notebook/campaign/runtime changes to
   `3.10_to_3.14` before uploading the notebook. Setup clones only that branch
   with `--single-branch --branch 3.10_to_3.14`, keeps it checked out, and records
   its exact fetched commit for all subsequent identity checks. There is no
   commit/tag override or detached checkout. Verify the expected model digest
   and dataset hash. `FULL_RUN_ID` is generated automatically for a fresh run;
   supply an existing ID only for intentional partial resume.
2. Keep `RUN_FULL_BENCHMARK=False` while running setup, preflight and pilots.
   The initial server uses one model slot. Context reduction and CPU offload
   are forbidden.
3. Run the fresh sequential canary. The default sample is three conversations
   selected by the original seed-shuffled runner and five ordered questions
   **per conversation** (normally 15 questions). Each sampled conversation
   replays every historical turn. The separate default smoke mode does not
   provide representative timing evidence for this production gate.
4. The enabled throughput pilot repeats exactly the same IDs in a fresh
   namespace with two isolated conversation workers and two model slots.
   Each conversation still processes turns and questions serially. Predictions,
   outcomes and evidence/evaluation decisions must match the sequential run.
   Provider telemetry must show two actually overlapping requests. A failed,
   changed-prediction or slower pilot retains/restores one-worker execution.
5. Inspect the saved timing and budget reports. Only after a passing estimate,
   set `CONFIG['RUN_FULL_BENCHMARK'] = True` immediately before the full-run
   cell. It rechecks live repository/model/dataset identities, unchanged
   effective config, full selected IDs, resources and elapsed session time.

Canary and pilot namespaces include a unique session suffix and **never
resume**. An existing canary root fails before launch. Full campaigns resume
only a partial campaign with an exactly matching identity/config/selected-ID
marker, campaign dataset hashes and the canonical child manifest checks.
Completed campaigns require a new full-run ID. Explicit zero question limits
prevent environment defaults from accidentally restricting the full dataset.

## Runtime estimate and resource limits

Conversation timing records separately measure full history replay, ordered
QA work and total worker time. The projection counts replay once per
conversation. Unsampled conversation histories/QA use the slowest measured
seconds per turn/question, with exact dataset turn/question counts. Scheduling
efficiency is measured from execution wall time divided by summed worker
time, bounded by the tested worker count and the longest ordered conversation.
Two workers do not imply a 2x speedup.

The gate includes campaign startup, scaled evaluator finalization, a measured
canary report/validation/ZIP probe, a minimum 20% safety margin and a minimum
900-second final artifact reserve. Actual elapsed environment inspection,
dependency installation, model pull/load, preflight, both pilots and server
restoration count against the session budget. Missing, stale, failed or
nonfinite measurements cannot pass. The campaign including artifact headroom
must fit **10.5 hours**, and elapsed plus remaining work must fit **12 hours**.
Rejected estimates print projected campaign/session hours and the shortfall.

The hard observed ceiling is **12 GiB per GPU**. Both GPUs must hold the exact
model, with full GPU residency and unchanged context. RAM planning uses
**19 GB decimal**, keeps at least **3 GB available**, and also respects a
stricter measured cgroup limit. RAM measurement failure blocks execution.
Resource violations stop the campaign/model and leave recoverable checkpoints.

GPU/RAM safety sampling remains at most two seconds apart. Progress renders
every 30 seconds by default and consumes only appended checkpoint bytes.
Samples include used GPU memory/utilization, system available memory,
process-tree RSS and cgroup usage/limit where available. They are appended,
flushed and synced immediately, including a sample that trips a limit.

## Artifacts and local evidence

The existing scoring, failure taxonomy, component/retrieval/reflection diagnostics,
selected/checkpoint reconciliation, duplicate checks, identity/required-file
validation, secret scan, human report, manifest and final ZIP remain. Added
artifacts include `run_identity.json`, `resource_samples.jsonl`, `progress.jsonl`,
child `timing.jsonl`, `session_timing.jsonl`, `timing_canary.json`,
`throughput_pilot.json`, `budget_gate.json` and ZIP packaging timing.
Optional plots are off by default because the validator does not require them.
Paired baseline and semantic metrics remain opt-in and count toward timing.

Focused synthetic tests exercise fresh/stale identities, replay deduplication
and extrapolation, elapsed-session gating, incremental/partial JSONL reads,
durable failure telemetry, concurrency guards, ordered isolated conversation
execution, notebook validation and ZIP creation. Existing campaign, comparison,
artifact and LoCoMo tests are also run. No full Kaggle campaign is launched.

The local machine has a single 4 GiB RTX 3050 Laptop GPU, so it cannot measure
the requested two-T4 model pilot or prove the 12 GiB-per-GPU resource envelope.
Fake-provider/runtime checks prove control flow and invariants, not GPU
throughput. Safe optimization alone has **not established single-session
feasibility**; the full-run flag remains off until fresh Kaggle evidence passes.

## Inspection after the request to stop testing

No further tests or benchmark executions were run. Notebook inspection found
manual commit/run-ID stops, detached checkout, an unnecessary registry pull
for an existing model, and repeated optional spaCy model loads. Setup now
checks out only the requested branch and captures its commit; fresh full-run
IDs are automatic. Existing Ollama caches are discovered for production too,
cached models skip pulling and the download-space guard, missing models have
a bounded download, and model residency is retained for the session. spaCy
is loaded once per worker thread, preserving its existing extraction policy
and missing-model fallback.

The reference projection still does not support a 12-hour promise. Halving
21h23m gives about 10h42m before setup and final artifacts; adding the existing
20% safety margin already exceeds 12 hours. The 10.5-hour campaign gate is
stricter still. These changes may remove overhead, but their speedup has not
been measured. Actual two-slot model/context residency and representative
timing must pass before the notebook permits the full run.
