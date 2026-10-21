# Kaggle LoCoMo session budget

## Current execution contract (21 October 2026)

The Goal 4 handoff and [campaign runbook](benchmarks/RUNBOOK.md) supersede the
operating instructions below. Historical timings remain observations for their
recorded source/model/configuration; they do not admit the repaired runtime.
Use reviewed `dev` with an explicitly agreed `PINNED_COMMIT_SHA`, an immutable
executed-notebook input, one shared pinned 27B model and one worker initially.
Full execution defaults off. Unknown/one backend slot retains one worker;
HTTP request overlap is not evidence of concurrent model execution. Optional
repeated pilots require equal native outcomes, an unambiguous measured wall win,
a passing budget and successful backend restoration. LoCoMo CPU overlap with
serialized inference is labeled separately. Report the actual sampling gaps;
a configured cadence is not a guarantee. Live quota, fresh source/config/model/
data identities, 20% margin and a 900-second reserve are mandatory. Current
LoCoMo and learned-policy candidates remain unpromoted pending measured gates.

The supervised October 5 run at `c6e11648` completed the fresh 15-question
sequential canary in 415.5 seconds. Its full-history, full-coverage projection
was **16.9800 hours for the campaign** and **17.3781 hours for that session**,
including the 20% safety margin and 15-minute final artifact reserve. The
two-worker pilot completed its questions but was refused because predictions
or evaluation decisions differed from the sequential reference. It therefore
cannot authorize concurrent production. These are conservative projections
from three sampled conversations, not a completed full benchmark measurement.
A fresh session does not make the current configuration fit twelve hours;
HotpotQA and GoEmotions still need their own representative measurements.

The same run exposed a remaining full-launch checkout guard that counted
missing uninitialized external submodule directories as tracked edits. Setup
and full launch now use the same submodule exclusion; real owned-source edits
still block both stages. The timing gate remains unchanged and refuses this
run independently of the checkout correction.

The canonical `PRIMA_Kaggle_Benchmark.ipynb` preserves all 1,986 questions,
`prima_full`, `normal_prima_admission`, full historical replay,
`flush_before_question`, seed 13, 8192 context tokens, 512 output tokens and
the configured retry policy. The local LoCoMo dataset has ten conversations,
5,882 historical turns and SHA-256
`553cd5a15e25f2ceccc6ed185221eba645080c93e5b91087560a91aa5961f365`.

The supplied reference run reached 698/1,986 questions after 27,049 seconds
with 49,912 seconds remaining: roughly 21h23m projected, before setup and
final artifacts. Its fixed-path, resumable canary is not reliable fresh
throughput evidence. Its saved source, output and timestamps are now directly
inspected in [reference analysis](kaggle-reference-analysis.md). These are
historical observations, not a reproduced run of the changed code. The displayed 19.47
GiB was disk space, not a measured RAM peak.

## Running the notebook

The saved `notebook0c24a612be.ipynb` run fetched `e524609`, pulled the pinned
model, passed the 8192-context two-GPU preflight (9,547/9,721 MiB VRAM), then
stopped at the first sequential canary with `Cgroup RAM reserve violated`.
It contains no completed canary timing or full-run results. The raw cgroup
guard counted filesystem cache as unreclaimable memory; the saved notebook
does not include the counters needed to establish how much cache caused this
particular failure.

Warm-up, preflight and campaign monitoring now record the same Linux cgroup
v1/v2 counters. Clean disk cache on both the active and inactive reclaim lists
receives credit, bounded by filesystem cache minus shared/tmpfs memory.
Dirty/writeback pages and locked memory receive none. Missing or invalid counters
retain the raw-usage guard. The 19 GB planning
cap and 3 GB reserve still apply to the working set, and failure messages give
raw usage, cache credit, working set and limit. Counter definitions follow the
[Linux cgroup documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory).
These accounting changes do
not establish a 12-hour completion time; the fresh canary must still pass.

The supervised October 5 run exposed a second accounting error: during model
loading, clean file pages moved from the inactive to the active list. The old
inactive-only subtraction increased the apparent working set to 16.858 GB
after 22.569 seconds and aborted loading despite 30.657 GB of system available
RAM. The failure sample recorded 28.999 GB total cgroup usage, 27.799 GB file
cache, 0.740 GB shared memory and only 24,576 dirty bytes. After the abort,
`memory.stat` still showed 14.941 GB active file cache versus 0.799 GB anonymous
memory. This was a notebook guard failure, not evidence of a kernel OOM.
The shared guard now credits both clean file reclaim lists while preserving
the 16 GB working-set ceiling, 3 GB reserve and 12 GiB/GPU ceiling.

Upload the updated canonical notebook and start a fresh Kaggle session rather
than continuing cells in the failed session with old imports. For any further
failure, save `/kaggle/working/prima_outputs` together with the notebook so the
resource and timing measurements remain available.

The notebook now defaults to an isolated two-example smoke mode. See
[three-benchmark smoke validation](kaggle-smoke-validation.md) for its exact
steps and limitations. The production timing steps below require
`SMOKE_MODE=False` in a new session.

1. After human review, use the integrated `dev` commit and enter its exact
   `PINNED_COMMIT_SHA`; no automatic publication or implicit latest-head update.
   Keep the model digest/context/output fixed. Pin the uploaded notebook input.
2. Keep full execution off. The current LoCoMo candidate also requires a hashed,
   reviewed promotion manifest with matched answerable official-F1/adversarial,
   citation/failure/read-only-QA and integrated resource/time evidence.
3. Run a fresh one-worker canary on three complete histories and ten ordered
   questions each. Tiny smoke is a separate two-example wiring check.
4. Optional two-client trials use fixed native IDs and repeated timings.
   Slower/noisy results, unknown backend slots, outcome changes or failed budgets
   retain one. CPU overlap with a single inference queue is an explicit LoCoMo
   experiment. Never infer model slots from requested settings or HTTP overlap.
5. Restore/re-preflight the selected backend. Recheck actual source/model/data/
   config, sampling coverage, remaining session time and live quota. Count setup,
   downloads, pilots, restore, reports and export; reserve is counted once.

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

GPU/RAM sampling uses monotonic deadlines and records actual gaps; missed two-second deadlines fail closed rather than being described as a cadence guarantee. Progress renders
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
