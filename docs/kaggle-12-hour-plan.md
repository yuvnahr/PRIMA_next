# Plan: complete each benchmark in one 12-hour Kaggle session

Use one separate two-T4 session for LoCoMo, HotpotQA and GoEmotions. The goal
is every source example scored and packaged within that session. Dataset
coverage, native evaluation, evidence checks, maintenance barriers and durable
checkpoints remain required. A timeout or incomplete selection is not success.

The current 27B setup has not demonstrated this outcome. The supplied LoCoMo
reference projected 76,961 seconds (about 21h23m). Dividing that by two is
insufficient once setup, validation and the 20% safety margin are included.
No unmeasured speedup is credited in the launch gate.

## Budget and acceptance criteria

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
   missing-cache blocker. Install only needed runtime/benchmark dependencies
   in a later setup cleanup, without changing their supported behavior.

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

3. **Improve LoCoMo scheduling without changing its conversation state.**
   Keep turns/questions serial within a conversation and isolate its runtime.
   Run independent conversations on the already supported two-worker path.
   Schedule expensive conversations first using source history/question
   counts and measured costs; keep manifest/source ID order unchanged.
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

5. **Choose the model topology from evidence.** First attempt the pinned
   27B model with the preceding fixes. If it still cannot meet the gate, the
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
   GoEmotions must cover differing text/output sizes. Use conservative cost
   estimates for unsampled cases and record the sampling limitations.

## Model identity blocker

The configured reference digest starts `22130167c4c2`. The public
`qwen3.8:27b` tag currently lists a different digest prefix, `e118e4d12a70`.
Pulling a mutable tag is therefore not proof of obtaining the pinned model.
Use the matching cached snapshot for the original experiment, or independently
verify and pin the new full manifest digest and record a new experiment before
running all three benchmarks. Never automatically trust the observed digest
after a mismatch. This is separate from timing feasibility.

## Evidence required for a green light

For each benchmark: dataset/split hashes and complete source IDs/count;
repository commit; actual model/backend/context identity; real canary and
pilot predictions/checkpoints; observed concurrency; inference/replay/
maintenance/retrieval timing; sampled GPU/RAM peaks; conservative projected
complete campaign and remaining session time; passed budget/resource gates.
Final reportability additionally requires complete successful full coverage,
native metrics/artifacts and a validated ZIP.

The first implementation batch includes setup/coverage fixes, thread-local
spaCy loading, an execution-ID event lookup used by runtime trace assembly,
and cached float32 normalized retrieval vectors invalidated when embeddings
change. Full event retention, cosine calculation and stable result ordering
are preserved. Scheduling improvements, HotpotQA/GoEmotions concurrency and
the smaller-model replica topology remain subsequent work; they are not
claimed as implemented or measured.

Current status: local control-flow checks from the earlier work passed, but
there is no connected Kaggle runtime/cache or real GPU timing evidence here.
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
- [Qwen3.8 published tags](https://ollama.com/library/qwen3.8/tags).
- [Qwen3.5 9B model details](https://ollama.com/library/qwen3.5:9b): published
  Q4_K_M artifact size is 6.6 GB; that is a weight-file size, not a measured
  total runtime VRAM allocation or a speed guarantee.
