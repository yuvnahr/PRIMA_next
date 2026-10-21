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
