# Phase 11 — HotpotQA Canonical Runtime Migration

## Outcome

HotpotQA now measures canonical runtime profiles instead of a benchmark compatibility pipeline. Every supplied sentence enters as typed `document_ingestion` with `ingestion_only`; every question enters as typed `factual_qa` using `model_only`, `simple_rag`, or `prima_full`. Both call `PrimaRuntime.execute()` directly.

The workflow remains the sole production orchestrator. No benchmark code was added to production packages, no gold data or historical result artifact was changed, and official HotpotQA answer, supporting-fact, and joint metrics remain intact. `Draft.png` remains the target architecture, not a claim that every block ran in every profile.

## Paired execution and context attribution

- `--paired` resolves one sample selection and seed, then runs all three profiles with identical provider, model, generation configuration, and selected IDs.
- Profile differences are declared route capabilities; benchmark concepts do not enter production.
- Parallel samples use isolated runtime state/memory while sharing one injected heavyweight model client.
- `distractor` is supplied distractor context.
- `official_retrieved` is the benchmark authors' official retrieved context; `fullwiki` remains a legacy CLI/dataset alias. It is not open-domain Wikipedia retrieval.
- `oracle` filters with supporting-fact gold only for diagnostics and records `headline_eligible=false` plus `oracle_diagnostic_only=true`.

## Diagnostics and official scoring

Per-item artifacts take retrieval hops/counts, reflection interventions, evidence IDs, title/sentence provenance, latency, model calls/token usage, and stop reason from `PrimaResponse` diagnostics.

Official answer EM/F1, supporting-fact precision/recall/F1/EM, and joint EM/F1 are unchanged. Ordinary wrong answers are scored as `ANSWER_SCORING_FAILURE`, never runtime failures. Reachable Hotpot subcodes are `RUNTIME_FAILURE`, `INGESTION_FAILURE`, `RETRIEVAL_MISS`, `REASONING_STOP_BUDGET`, `SUPPORTING_FACT_PROJECTION_FAILURE`, `ANSWER_PARSE_FAILURE`, and `ANSWER_SCORING_FAILURE`. Summaries reconcile processed, completed, failed, scored, and categorized totals.

## Crash safety and resume

The obsolete local checkpoint module and ignored `checkpoint_every` option were removed. The Phase 10 store now fsyncs each terminal item immediately, suppresses duplicate IDs, retains partial status on interruption, validates exact resume compatibility, and writes the authoritative complete manifest last.

## Verification

Focused implementation checks:

```text
venv\Scripts\python.exe -m pytest -q tests/benchmarks/test_common_artifacts.py tests/benchmarks/test_hotpotqa.py tests/benchmarks/test_hotpotqa_qol.py
31 passed in 3.29s

venv\Scripts\python.exe -m ruff check benchmarks/common/contracts.py benchmarks/hotpotqa tests/benchmarks/test_hotpotqa.py tests/benchmarks/test_hotpotqa_qol.py
All checks passed!
```

Final repository quality gates were run manually by the user and reported passing on 2026-08-10 under Python 3.10.11. Exact gate output was not captured by Codex, so this report does not invent command totals.

## Changed files

- `benchmarks/common/contracts.py`
- `benchmarks/hotpotqa/{README.md,adapter.py,config.py,console.py,data_sources.py,evaluate.py,experiment.py}`
- removed `benchmarks/hotpotqa/checkpoint.py`
- `tests/benchmarks/{test_hotpotqa.py,test_hotpotqa_qol.py}`
- `docs/implementation/{STATUS.md,phase_11_hotpotqa_canonical_runtime.md}`
- `pyproject.toml` removes the obsolete checkpoint-module lint override.

## Deferred risks

- LoCoMo and GoEmotions still own later migration phases; the three benchmarks must not yet be presented as one complete-wrapper measurement.
- Hotpot uses supplied benchmark context. This phase makes no open-domain retrieval claim.
- Model-quality scores were not regenerated; this phase repairs the measured path and artifact truth.

The pre-existing `benchmarks/locomo-v2/external` submodule marker remains untouched.
