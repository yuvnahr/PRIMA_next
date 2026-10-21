# LoCoMo Benchmark Integration

This folder contains PRIMA's architecture-only integration layer for the LoCoMo benchmark.
The official LoCoMo repository is kept isolated under `external/` as a git submodule.

## Folder Purpose

- `config.py`: benchmark-local configuration.
- `loader.py`: locates and loads raw LoCoMo JSON.
- `adapter.py`: converts raw LoCoMo records into shared `Conversation` objects.
- `runner.py`: reuses the benchmark-agnostic runner contract.
- `evaluate.py`: computes metrics from runner outputs only.
- `outputs/`: benchmark-local raw, processed, metric, plot, and log artifacts.

## Execution Flow

```text
LoCoMo JSON
-> loader.py
-> adapter.py
-> Conversation
-> generic agent runtime
-> runner.py outputs
-> evaluate.py metrics
```

The adapter and generic runner use benchmark-neutral conversation objects. The canonical
`experiment.py` and retrieval development entry point use the actual PRIMA runtime.

## Expected Outputs

Benchmark outputs should be written under `outputs/`:

- `raw/`: raw runtime outputs.
- `processed/`: normalized intermediate artifacts.
- `metrics/`: evaluation summaries.
- `plots/`: benchmark-specific figures.
- `logs/`: benchmark logs.

## Evidence and scoring contract

Canonical experiment question/metric schema 2.0 separates all-hop candidate,
integrator-accepted, actually serialized prompt, and model-cited evidence. Runtime
`output_data.evidence_audit` version 1.0 is available without full diagnostic tracing;
the legacy `retrieval` view explicitly covers only the last retrieval call. Missing
old-record observations remain unavailable rather than zero recall.

The evaluator reports legacy normalized EM/SQuAD-style F1 and smoothed sentence BLEU
alongside the pinned official LoCoMo core score and scorer identity. Structured
category-5 abstention and the official raw-text refusal score are separate metrics.
Correctness and annotation-based evidence flags are independent; recall denominators
separate answerable/adversarial and unannotated populations.

The adapted official scorer retains snap-research/LoCoMo attribution and its
[CC BY-NC 4.0 license](official_scoring_LICENSE.txt). Core scoring requires no
BERTScore import or model download. See the
[implementation handoff](../../docs/locomo-review-remediation.md) for exact versions,
compatibility aliases, fixtures, resume rules and validation limits.

## Historical replay and CPU retrieval candidates

Replay preserves source time, numbered session, turn, speaker and conversation identity.
Malformed historical time fails before admission. `experiment --include-captions` appends
only supplied `blip_caption` strings; the default is off. Caption policy and replay text
have a separate representation fingerprint while the original dataset hash is retained.

The stable-hash 64-dimensional default remains unchanged. Learned selection requires an
immutable model revision and native dimension (384 for MiniLM). Auxiliary loaders use
explicit devices and staged local files only. The typed settings, exact model revisions,
custom snapshot manifest format and Person 2 staging requirements are in the handoff.
Select a complete measured candidate with `RuntimeConfig.model_validate(candidate)` from
`tests/fixtures/locomo_review/goal2_retrieval_candidates.json`, then pass it as
`runtime_config=` to `run_locomo_experiment`. Changed identities require fresh runs/index
rebuilds; incompatible checkpoints cannot resume.

Reproduce the frozen retrieval-only development ladder from the repository root with a
fresh output directory and the exact cached MiniLM revision already staged:

```powershell
python -m benchmarks.locomo.retrieval_development --dataset benchmarks/locomo/external/data/locomo10.json --manifest tests/fixtures/locomo_review/goal2_development_manifest.json --candidates tests/fixtures/locomo_review/goal2_retrieval_candidates.json --output benchmarks/locomo/outputs/goal2/development-new
```

This replays complete development histories through controlled ingestion and performs no
answer generation. Five CPU candidates were measured; the pinned cross-encoder is pending
local artifacts. Reports include stage recall, separate stage costs and replay reuse.
All candidates remain unpromoted. The opt-in real-cache test is
`PRIMA_RUN_LEARNED_EMBEDDING_SMOKE=1` with
`pytest -q tests/test_production_embedding_smoke.py -k pinned_cpu`; ordinary tests use fake
models and do not download artifacts.

## Grounded answer attempts

Canonical acquisition preserves the original question in supported follow-ups and defers
generation to the workflow. Eligible low-overlap evidence can receive one bounded attempt
without being labeled sufficient; the model can still abstain. Shared typed call/deadline
controls apply before generation and provider retries, with answer time reserved before
another retrieval hop. Empty evidence, contradiction, hard errors and exhausted budgets
prevent speculative calls.

Responses and question records add `answer_attempt_permitted`, `answer_attempt` and
`request_budget`; old observations remain null. The evidence/scorer contract is unchanged.
`answer_contract_version=1.0` invalidates incompatible experiment resumes. See the
[Goal 3 handoff](../../docs/locomo-review-remediation.md#goal-3--bounded-grounded-attempts-and-question-preserving-follow-ups)
for defaults, consumer fixtures, exact checks and pending quality/timing gates.

## Goal 4 policy and review handoff

Factual retrieval excludes only verified fact-free autogenerated cluster descriptions
before strategy ranking and final top-k. Factual imported semantic documents remain
eligible, and source lineage remains available for audit. Historical replay writes one
complete admission log per conversation under `store/admission/`, including rejected
turns, component scores where the policy computes them, threshold, reason and created
IDs. QA does not write memories.

The safe explicit configuration is `stable64_normal_admission_reflection_off` in
`tests/fixtures/locomo_review/goal4_final_candidates.json`; the pinned native MiniLM384
configuration there is **unpromoted**. `goal4_policy_matrix.json` freezes six CPU-only
stages on the six complete development histories and 116 paired QA IDs. The normal
admission stage has 2,500 memories and returned annotation recall@5 of 0.333333 over
91 annotated answerable questions. Controlled all-turn ingestion has 3,329 memories
and 0.291209; source-factual retention has 2,581 and 0.331502; supplied captions
have 2,474 and 0.356227, with a whole-conversation confidence interval crossing zero.
These are retrieval proxies, with no answer-model quality or Kaggle resource result.
No threshold or policy was promoted. Both no-model reflection stages executed zero
interventions; benchmark reflection defaults off.

Reproduce the frozen matrix with a fresh output directory after staging its pinned
native CPU model:

```powershell
python -m benchmarks.locomo.policy_development --dataset benchmarks/locomo/external/data/locomo10.json --split tests/fixtures/locomo_review/goal2_development_manifest.json --controls tests/fixtures/locomo_review/goal4_policy_matrix.json --output benchmarks/locomo/outputs/goal4/policies-new
```

The [four-goal handoff](../../docs/locomo-review-remediation.md#goal-4--final-policy-implementation-and-review-package)
records the complete stage ledger, admission/QA producer fixture, schema versions,
artifact hashes, selection gate, rollback and Person 2 integration instructions.
Full LoCoMo answer-quality and hardware acceptance remains after human review: all
10 conversations and 1,986 questions exactly once on a frozen, resource-admitted run.
