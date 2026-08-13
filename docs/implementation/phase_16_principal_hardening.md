# Phase 16 — Principal-Engineer Hardening

## Outcome

The canonical runtime now fails closed around one authoritative route plan. Caller
metadata cannot set routing, policy, retrieval, budget, reranker, compression, or
tool controls; immutable `ExecutionOptions` owns those values. Tool requests return
the typed `actioned` outcome and report `tool_executor` only when its action phase
actually completes.

Diagnostics distinguish retrieval calls from returned results, workflow reflection
from reasoning/query recovery, and accepted corrections from attempted reflection.
Planned, enabled, executed, not-executed, and route-unselected components have
separate meanings, and executed reranker/embedding/affect/provider identities are
reported without hidden backend fallback.

Campaign preflight and manifests use the same effective task-specific generation
configuration as execution. Factual QA is explicitly JSON-schema constrained before
preflight, QA context/output/safety budgets must fit the provider window, and affect,
embedding, retrieval, and reranker environment values resolve once into a typed,
fingerprinted runtime configuration.

Paired comparison now validates dataset hash, context variant, headline eligibility,
selection/order, model/revision, generation, seed, context, and retry invariants. Its
bootstrap resamples paired records and recomputes the requested benchmark-native
metric. Failed, parse-failed, and unscored cases are excluded and counted rather than
silently assigned zero. HotpotQA and LoCoMo evidence metrics use only generation-cited
evidence. GoEmotions paired transforms replay the identical cached raw response and
record its SHA-256 hash while reusing classification runtimes per worker.

Checkpoints are now the durable item source of truth. Prediction/failure JSONL files
are derived mirrors reconciled deterministically at initialization, and in-memory
indexes remove per-append full-file scans. Heavy memory maintenance is batched until
a declared flush boundary; graph synchronization is lazy and subsequent writes stay
incremental. The maintenance queue remains process-local and only terminal failure
records are durable.

## Verification

- Focused hardening campaigns: 104 passed, then 22 passed, then 47 passed; corrected
  fail-closed direct-workflow tests: 3 passed across their focused reruns.
- Owned-source compileall: passed.
- Complete pytest pass: 334 passed, 1 optional skip, and 3 stale direct-workflow tests
  failed because they supplied no trusted route. Those tests were corrected to inject
  explicit routes/configuration and all three passed focused reruns. The complete
  suite was not rerun in accordance with the one-final-CI instruction.
- Ruff: initial nine formatting/unused-name findings corrected; repository recheck
  passed.
- mypy: initial four typing findings corrected; recheck passed for 234 source files.
- Bandit: passed with only existing informational `nosec` warnings.
- Production-only Xenon: passed.
- Fake-provider campaign `campaign-9d98edf3-d542-4dfb-a63a-89a8ae31fde5` completed all
  modes with 4/4 successful requests and maximum active inference of one.
- SIGINT and SIGTERM subprocess interruption/resume: 2 passed; checkpoint IDs remained
  unique.
- Smoke artifact audit: zero credential/developer-path matches; every checkpoint and
  prediction mirror had matching unique case counts.

## Decision and residual risk

GO for the fixture-validated code boundary. The expensive full-dataset GPU campaign
remains NO-GO until real model, revision, endpoint, dataset, device routing, and
output fields are supplied and pass preflight. The fake provider establishes
orchestration and reliability, not model quality. GPU boundary snapshots are not peak
measurements, and process loss can discard queued maintenance events that have not
reached their terminal failure record.
