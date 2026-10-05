# Plan: complete each benchmark in one 12-hour Kaggle session

Use one separate two-T4 session for LoCoMo, HotpotQA and GoEmotions. The goal
is every source example scored and packaged within that session. Dataset
coverage, native evaluation, evidence checks, maintenance barriers and durable
checkpoints remain required. A timeout or incomplete selection is not success.

The current 27B setup has not demonstrated this outcome. The saved old notebook
has now been inspected: see [reference analysis](kaggle-reference-analysis.md).
Its partial LoCoMo run projected 76,961 seconds (about 21h23m). Dividing that by two is
insufficient once setup, validation and the 20% safety margin are included.
No unmeasured speedup is credited in the launch gate.

## Redesign after the measured LoCoMo pilot (2026-10-05)

The current pinned 27B profile projects a 16.98-hour campaign and a 17.38-hour
session from the representative pilot. Both fail admission. Its model already
resides entirely on the GPUs; increasing occupied VRAM is not an optimization.
The two-worker pilot failed prediction/evaluation equivalence and is not an
accepted production configuration. HotpotQA and GoEmotions remain unmeasured.

Before another hardware run, retain the saved sequential/concurrent pilot
records and identify the actual differing fields. `validate_pilot` now reports
case IDs and field names, without copying answer text into the exception. Every
existing equality, overlap and resource requirement remains enforced.

Campaign provider telemetry now sums native model-load, prompt-evaluation,
decode and total durations, with a sample count for each field. Cache hits are
excluded, missing measurements remain absent, and sums from overlapping
requests are explicitly distinguished from wall time. These are diagnostics,
not speedups or permission to reduce the budget projection.

Use the measured dominant cost to select the next change: fix any shared-state
defect demonstrated by the pilot diff; otherwise evaluate isolated model
replicas with a uniformly pinned smaller model as a new experiment. Do not
spend another production session repeating a failed unchanged profile. Keep
complete histories/examples, native scoring, 8192 context and all memory/time
guards. No profile is ready until each benchmark's real representative pilot
passes those guards and leaves room for setup and final artifacts.

The saved Version 1 archive was downloaded and independently verified against
SHA-256 `e77921d2e9debd130920b322ec6e005fcd52ac9cb71c67add609b6fd97d9d236`.
Direct comparison found ten differing cases, primarily candidate evidence IDs,
two candidate-recall/failure-category changes, and one prediction change.
Native sequential QA records contain seven model calls: 84.307 seconds prompt
evaluation, 15.466 seconds decode and 0.014 seconds model loading, against
415.502 seconds total campaign wall time. Those QA sums do not include every
campaign phase and must not be treated as a complete wall-time breakdown.

The archive also exposes a correctness blocker previously hidden by successful
campaign status: conv-42 admitted 561 memories but completed only 128
maintenance events, with 433 `maintenance_queue_full` failures. Queue admission
currently uses nonblocking `enqueue` in the shared workflow, while synchronous
history execution can starve the asynchronous worker until a barrier. Fix
bounded producer backpressure and require completed maintenance in canary
validation before another readiness claim. Do not enlarge an unbounded queue
or omit failed work to improve timing. Investigate the maintenance worker's
embedding context too: it is started outside the runtime's request-scoped
embedding-pipeline context, and maintenance calls `get_embedding_pipeline()`.
The observed evidence changes alone do not establish which defect caused them.

The shared workflow now awaits bounded backpressure when its maintenance queue
fills instead of dropping events. The explicit nonblocking supervisor API
retains its saturation behavior for callers that choose it. Runtime worker
startup and maintenance barriers bind the owning runtime's embedding context;
background tasks must not inherit the global default pipeline. The notebook
now rejects a canary whose final conversation diagnostics report incomplete
maintenance, regardless of otherwise successful campaign status. Local
regressions cover queue pressure, owning-pipeline identity and rejection of a
corrupted final-maintenance record. This repair processes previously dropped
work, so the old 16.98-hour projection is not an accepted timing measurement
of the repaired profile and cannot certify its speed. Re-measure after the
remaining performance work, preserving all admission and memory requirements.

### Bounded CPU optimization and reproducibility follow-up

Local profiling of three maintenance batches over 128 LoCoMo source texts
recorded 79,127 calls to the exact substring keyword-overlap calculation.
The shared function now uses a standard-library LRU capped at 65,536 entries,
keyed by immutable keyword tuples. Direction, duplicate handling and substring
semantics are unchanged; changed keyword lists produce different keys. All
graph pairs, edges and weights are still calculated or retrieved exactly.

A local comparison repeated five full graph rebuild/index passes over those
same 128 source texts, three times per implementation. Median time was
0.602 seconds uncached versus 0.231 seconds cached (2.60x for this component),
with identical 128 nodes and 1,108 edges/weights/metadata. Cache telemetry
reported 351,711 hits and 14,049 misses. This microbenchmark does not measure
Kaggle throughput, total maintenance cost, peak campaign RAM or model inference;
its speedup is not applied to the 12-hour admission projection.

Graph traversal now resolves equal-weight neighbor ties by node insertion
order, rather than set iteration over random memory IDs. Evolution orders
community parents by repository insertion order before constructing summaries,
concepts and lineage. Regression checks reverse community iteration and alter
edge weights to verify stable summaries/lineage and correct cache invalidation.
These correct arbitrary ordering, so fresh pilot identities and measurements
are required; they do not establish that every saved pilot discrepancy is
resolved. The full native equality guard remains mandatory.

## Budget and acceptance criteria

### Explicit single-GPU replicas (implementation, not timing evidence)

The notebook now defaults to a new Qwen3.5 9B Q4_K_M experiment pinned to
`56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`.
Two Ollama servers share one weight cache, each exposing one inference slot
and seeing only its assigned T4. Sequential warmups verify the GPU allocation
delta separately; every server must retain the pinned model at 8192 context
and full GPU residency. Campaign workers use sticky replica routing with a
one-request semaphore per server. Actual send overlap, not time spent waiting
on those semaphores, supplies concurrency telemetry. Pilot transitions retain
warm replicas rather than downloading or reloading them again.

Both server identities are checked during campaign preflight, live resource
sampling and final artifact validation. Legacy shared-server configuration
remains explicit. Local routing, wrong-GPU and second-replica residency checks
are synthetic software regressions, not hardware measurements. All three
benchmark pilots and complete runs still need real Kaggle time/memory evidence;
no model-size ratio or assumed twofold speedup enters the admission gate.

| Work | Target per session |
|---|---:|
| Cached environment/model setup | 15 minutes |
| Real preflight, representative timing and concurrency validation | At most 75 minutes |
| Projected complete campaign, native scoring and measured packaging work | At most 8 hours before safety margin |
| Runtime safety margin | 20% of the campaign projection: 1h36m at the target |
| Additional final artifact reserve | At least 15 minutes |
| Unallocated headroom at those targets | 39 minutes |

These are targets, not observations. Existing gates remain stricter whenever
actual elapsed time leaves less headroom: campaign projection including
safety/reserve <=10.5h; elapsed session + projected remaining work <=12h.
Maintain <=12 GiB observed VRAM on each GPU, the 19 GB RAM planning cap and
at least 3 GB RAM available. Check any tighter measured cgroup limit too.

LoCoMo must improve about **2.67x** over the supplied projection to reach the
8-hour target. At 1,986 questions this is 14.5 seconds of campaign work per
question on average, including the cost of all histories; it is not a
14.5-second model-request timeout. For HotpotQA/GoEmotions, use the actual
pinned source count N: target overall throughput >=N/28,800 examples/second.
Do not substitute guessed dataset counts or tiny-smoke throughput.

## Implementation order

1. **Commit and publish the prepared setup/guard fixes.** Clone only
   `3.10_to_3.14`, capture its commit, generate fresh run IDs, reuse cached
   model weights, keep the model resident and retain all full-coverage gates.
   spaCy now loads once per worker instead of attempting a load repeatedly.
   Use existing caches for smoke; do not download large resources to hide a
   missing-cache blocker. The notebook now installs `requirements-kaggle.txt`
   for the native default profile, reusing compatible Kaggle packages without
   upgrades. HF/encoder backends or BERTScore explicitly select the full list.

2. **Remove repeated CPU work before increasing GPU workers.** Source
   inspection found that runtime trace assembly copies/scans the complete
   accumulated event history on every request. Add an execution-ID lookup
   to the existing event store and use it for the same trace records, retaining
   every event and its insertion order. Inspect graph scans and repeated
   float32 embedding conversion/norms next; cache only immutable derived
   values, invalidate on memory updates, preserve exact scoring/tie order.
   Do not replace full retrieval with approximate search or drop graph edges.
   Measure replay, maintenance, retrieval and inference separately in the
   actual campaign; prioritize the dominant measured cost.

   Source inspection of the old graph path found centrality sorting every
   adjacency list only to count neighbors. The implementation now reads
   exact adjacency degrees, caches descending neighbor-ID order with edge
   invalidation under a per-graph lock, and snapshots keywords once per
   rebuild. Every pair calculation, edge and maintenance operation remains;
   no performance factor is assumed from these source changes.

3. **Improve LoCoMo scheduling without changing its conversation state.**
   Keep turns/questions serial within a conversation and isolate its runtime.
   Run independent conversations on the already supported two-worker path.
   Schedule larger conversations first using source history/question counts;
   keep manifest/source ID order unchanged. The current heuristic gives each
   history turn/question equal weight. Refine it with measured replay/QA
   rates only if actual worker timings justify that change.
   Compare the same real QA predictions, evidence/evaluation decisions and
   maintenance outcomes against the sequential canary. Require observed
   overlapping provider requests and safe unchanged-context GPU residency.
   Extend the projection to include the actual worker assignment and the
   longest conversation, so balancing is not mistaken for a 2x guarantee.

4. **Enable measured concurrency for the other native runners.** HotpotQA
   already constructs isolated runtimes per independent case. GoEmotions has
   thread-local runtime/source state and a native parallel worker path.
   Start with two workers; update the campaign admission gate, notebook
   pilot and benchmark-specific equality checks together. HotpotQA equality
   must include answer/supporting facts and reasoning outcomes; GoEmotions
   must include raw response, baseline/final labels and bounded decisions.
   Retain one shared, bounded provider session and thread-safe checkpointing.
   Allow the faster setting only after its real pilot passes. Do not enable
   four slots merely because four CPU threads exist.

5. **Choose the model topology from evidence.** First attempt the available,
   newly pinned 27B model revision with the preceding fixes. If it still cannot meet the gate, the
   performance fallback is a uniformly pinned smaller model for all three
   benchmarks, such as Qwen3.5 9B Q4_K_M, with one independent model replica
   per T4 if measured residency fits. This changes the evaluated model and
   scores: record a new experiment/model digest, use the same model across
   all three, and never present its results as the original 27B experiment.
   Add explicit per-endpoint routing and replica resource checks before
   enabling that topology; the current both-GPUs-for-one-model check cannot
   simply be bypassed. Preserve 8192 context, full example/history coverage
   and native scoring. If even that profile misses the measured gate, choose
   a smaller uniform model/new experiment or report infeasibility; do not
   silently truncate workloads. No claim of 9B throughput is made yet.

6. **Authorize production from real measurements.** Run one representative
   canary and only the necessary candidate pilot per benchmark. Account for
   their full replay and all elapsed setup in the session clock. Repeated
   history replay during pilots can itself consume the deadline; stop
   unpromising profiles early and start a fresh session for a new model rather
   than spending the production window on many experiments. Two-example
   smoke only verifies mechanics. LoCoMo estimates separate full history and
   QA costs; HotpotQA must cover differing context sizes/difficulty and
   GoEmotions must cover differing text/output sizes. The notebook uses native
   seed sampling for independent cases and a slowest-observed-case projection
   floor, alongside measured wall throughput. This does not prove that every
   difficulty/length stratum is covered; record that sampling limitation.
   Representative canary/pilot campaign processes now stop at 75 minutes
   combined; the full-run gate still accounts for all actual elapsed time.

## Model identity and cold-start readiness

On 2026-10-05, a direct read of the public `qwen3.8:27b` registry manifest
returned SHA-256 `aaee06c39dcf2437cde036998d960e1fc1494b8191be7cc9657d01e509097813`.
Only small manifest/source metadata was fetched; no weights were downloaded.
Ollama's native pull saves these manifest bytes, and its model digest hashes
the saved manifest. The notebook now explicitly pins this available **new
experiment revision** for all three benchmarks. It retains 27B and Q4_K_M;
its scores cannot be identified as the old saved model experiment.

The old digest `22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643`
returned HTTP 404 when requested by digest. Reproducing that revision therefore
requires its matching attached cache and an explicit old expected digest.
The notebook reuses that cache without a registry pull. A fresh full-session
pull checks the public manifest against the configured expected digest before
transferring weights, then checks the local digest again after download.
Future tag drift fails early instead of wasting a 30-minute download window.

Attached `/kaggle/input` caches are read-only. The notebook creates a writable
cache in `/kaggle/tmp`, links the immutable blob files and copies small metadata.
This permits Ollama's metadata writes without duplicating model weights. Cached
setup is the 15-minute planning target; a cold pull can consume up to 30 minutes
plus installation/load time. Actual elapsed setup always reduces the full-run
budget, so the cached setup target is never credited to a cold-start session.

## Evidence required for a green light

For each benchmark: frozen selected input/split hashes and complete source IDs/count;
repository commit; actual model/backend/context identity; real canary and
pilot predictions/checkpoints; observed concurrency; inference/replay/
maintenance/retrieval timing; sampled GPU/RAM peaks; conservative projected
complete campaign and remaining session time; passed budget/resource gates.
Final reportability additionally requires complete successful full coverage,
native metrics/artifacts and a validated ZIP.

The committed implementation includes setup/coverage fixes, thread-local
spaCy loading, an execution-ID event lookup used by runtime trace assembly,
and cached float32 normalized retrieval vectors invalidated when embeddings
change. Full event retention, cosine calculation and stable result ordering
are preserved. LoCoMo now starts larger conversations first on its parallel
path and restores source order for conversation diagnostics. Its estimate
includes projected worker assignments, busiest-worker load and the longest
conversation, with the observed scheduling factor retained as another floor.
HotpotQA/GoEmotions now have two-worker notebook pilots through their existing
isolated native paths, benchmark-specific equality checks, resource/overlap
requirements and measured faster-setting selection. All these changes are
unmeasured here. Measured-cost scheduling refinement, difficulty/length
stratification and the smaller-model replica topology remain subsequent work.

Current status: local control-flow checks from the earlier work passed. The
old saved notebook provides partial historical LoCoMo timing and single-slot
placement evidence, but no timing/resource evidence for the updated code or
for HotpotQA/GoEmotions. No live Kaggle runtime is connected here. The user
requests source-based estimates without further tests; none were run.

The current notebook accepts a frozen SHA snapshot of the authorized selected
Kaggle input when an optional external dataset pin is absent. Explicit pins
(including LoCoMo's) are still enforced. Unpinned external full-run files are
rejected. Every source ID/count/hash and GoEmotions split/leakage check stays
required, with mutation between stages rejected. Missing optional HotpotQA/
GoEmotions external digests no longer make their full modes impossible.
The user requested no further local test runs; none are included in this
planning/commit step. The plan cannot honestly certify that every benchmark
finishes in 12 hours before the real measurements above exist.

## Primary references

- [Ollama concurrency, GPU placement, residency and KV-cache settings](https://docs.ollama.com/faq):
  parallel requests require additional context memory; a model fitting one GPU
  is normally placed on that GPU, otherwise spread across available GPUs.
  Cache quantization can change precision, so it is not a free equivalent
  optimization and is excluded from the first unchanged-model path.
- [Ollama generation telemetry](https://docs.ollama.com/api/generate): model
  load, prompt evaluation and decode durations/counts can distinguish costs.
- [Qwen3.8 registry manifest](https://registry.ollama.ai/v2/library/qwen3.8/manifests/27b).
- [Native Ollama pull implementation](https://github.com/ollama/ollama/blob/main/server/images.go)
  and [saved manifest digest](https://github.com/ollama/ollama/blob/main/manifest/manifest.go).
- [Qwen3.5 9B model details](https://ollama.com/library/qwen3.5:9b): published
  Q4_K_M artifact size is 6.6 GB; that is a weight-file size, not a measured
  total runtime VRAM allocation or a speed guarantee.
