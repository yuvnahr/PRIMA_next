# PRIMA-NEXT LoCoMo QA Benchmark Readiness Report

Date: 2026-07-18  
Branch: `locomo-eval`  
Commit inspected: `706076f633f5cdb82063f31ca23f2f24e6870e22`  
Production model: Ollama 0.32.1 / `qwen3.5:4b` / thinking off  
Validation limit: one LoCoMo conversation; no full benchmark executed

## Verdict

Functional benchmark readiness: **PASS**  
Repository cleanliness: **FAIL — pre-existing dirty external submodules and uncommitted audit fixes**  
Readiness score: **96/100**

The production QA pipeline executes end to end with Qwen 3.5 4B, explicit Nomic embeddings, raw representation, identity normalization off, a 60-candidate retrieval pool, deterministic generation, localized evidence metrics, and complete artifacts. The full benchmark was not executed.

## Repository validation

| Validation item | Result | Evidence |
|---|---|---|
| Repository is clean | **FAIL** | Parent worktree contains this audit's uncommitted changes. `benchmarks/locomo/external` and `benchmarks/locomo-v2/external` were already dirty and were not altered or reset. |
| Merged branch is internally consistent | **PASS** | `py_compile`, production imports, Ruff, focused tests, and `git diff --check` pass. |
| No stale benchmark code remains | **PASS** | Removed the dead `candidate_pool_multiplier` YAML key and obsolete branch instructions. |
| No duplicate benchmark implementations remain | **PASS** | One production `run_locomo_experiment`, `LoCoMoRunner`, and `LoCoMoEvaluator` implementation found. |
| No obsolete `retrieval-v2-debug` paths remain | **PASS** | Repository search returns no obsolete branch/path reference outside Git history. |
| All production imports resolve | **PASS** | Direct import check passed for loader, runner, experiment, evaluator, runtime, embedding pipeline, and LLM provider. |
| No provable merge dead code remains | **PASS** | The only proven dead configuration and stale debug instructions were removed; no speculative deletion was performed. |

## Benchmark pipeline validation

| Stage | Result | Evidence |
|---|---|---|
| Benchmark loader | **PASS** | Loaded `locomo10.json`; dataset SHA-256 recorded separately. |
| Conversation loading | **PASS** | One complete conversation replayed through the production adapter. |
| Question loading | **PASS** | 50-question controlled validation retained question IDs, categories, expected answers, and evidence IDs. |
| Retrieval | **PASS** | Dense, sparse, temporal, fusion, reranking, and final-selection traces were emitted. |
| Context construction | **PASS** | Final selected memories were injected into bounded answer context. |
| Prompt generation | **PASS** | Per-question production prompts were written to `prompts/prompts.json`. |
| LLM interface | **PASS** | 50 sequential Ollama generations completed without runtime errors. |
| Answer parsing | **PASS** | Raw, plain-text, JSON-answer, null-answer, and thinking-block extraction contracts pass. |
| Metric computation | **PASS** | EM, F1, BLEU, ROUGE-L, BERTScore, legacy memory hits, evidence-stage recall, category metrics, and failure taxonomy produced. |
| Artifact generation | **PASS** | Exactly one canonical artifact set was generated under the requested output root. |
| Report generation | **PASS** | QA report, run metadata, metrics, raw responses, parsed answers, prompts, and answers were generated. |

## Production and LLM validation

| Validation item | Result | Evidence |
|---|---|---|
| Qwen 3.5 4B supported by installed Ollama | **PASS** | Ollama 0.32.1 served `qwen3.5:4b` for the controlled run. |
| Thinking disabled | **PASS** | Native Ollama payload contains `"think": false`, equivalent to `ollama run qwen3.5:4b --think=false`; mocked payload contract and live metadata passed. |
| Prompt formatting | **PASS** | Production prompt artifact contains the expected instructions, question, and context. |
| Context injection | **PASS** | Retrieved final memories are present in the prompt artifact. |
| Deterministic decoding | **PASS** | Temperature 0 and seed 13 are forwarded in Ollama `options`; one worker is required. |
| Output parsing | **PASS** | Provider response and answer extraction tests pass. |
| Answer extraction | **PASS** | JSON `answer`, null/none, fenced text, and plain text handled. |
| Timeout handling | **PASS** | `PRIMA_LLM_TIMEOUT_SECONDS` is forwarded to every request; fault-injection contract passed. |
| Retry handling | **PASS** | `PRIMA_LLM_RETRIES=2` produces at most three attempts; fault-injection contract passed. |
| Explicit production embedding configuration | **PASS** | Nomic/raw/identity-off self-test passed with fingerprint `b6926f519e3351e97eeb477cd148843fab559514d92ed90053166e8f29b00f9a`. |

## Reproducibility validation

| Metadata | Result |
|---|---|
| Backend fingerprint | **PASS** |
| Benchmark metadata and dataset SHA-256 | **PASS** |
| Hardware metadata | **PASS** |
| Model/provider/thinking metadata | **PASS** |
| Python/runtime metadata | **PASS** |
| Seed | **PASS** |
| Benchmark version | **PASS** |
| Retrieval configuration | **PASS** — top-k, candidate-pool size, and fusion weights recorded; explicit embedding environment required by command |
| Generation configuration | **PASS** — temperature, top-p, decoding top-k, repeat penalty, seed, token limit, context size, and retrieved-memory count recorded per question |

## Evidence-backed ablations

All runs used one conversation only. No full LoCoMo run occurred.

### Memory-admission correctness fix

Changing only the embedding backend exposed a correctness defect: admission novelty used raw embedding cosine as a cross-backend absolute score.

| Configuration | Stored turns |
|---|---:|
| Stable, semantic, identity on, before fix | 539/629 |
| Nomic, semantic, identity on, before fix | 1/629 |
| Nomic, semantic, identity on, after fix | 597/629 |

The fix keeps semantic nearest-neighbor candidate lookup but computes novelty from lexical Jaccard overlap, making the admission threshold backend-independent. In the 50-question validation, annotated evidence storage recall was 0.9933.

### Fusion-weight ablation

Frozen variables: same conversation, first 50 questions, Nomic, raw representation, identity off, seed 13, top-k 5, Qwen 3.5 4B, thinking off. The only changed variable was fusion weights.

| Metric | Sparse-heavy 0.15/0.65/0.10/0.10 | Balanced 0.40/0.30/0.15/0.15 | Difference |
|---|---:|---:|---:|
| Candidate evidence recall | 0.7707 | 0.7707 | 0.0000 |
| Fusion evidence recall | 0.6827 | 0.7277 | +0.0450 |
| Final-context evidence recall | 0.3575 | 0.5542 | +0.1967 |
| All-evidence availability | 0.3400 | 0.5200 | +0.1800 |
| Average relevant final rank | 2.0000 | 1.6857 | -0.3143 |
| F1 | 0.0911 | 0.1526 | +0.0615 |
| BERTScore | 0.1257 | 0.1744 | +0.0487 |

Category F1:

| Category | Sparse-heavy | Balanced | Regression |
|---|---:|---:|---|
| 1 | 0.0598 | 0.1323 | No |
| 2 | 0.1117 | 0.1763 | No |
| 3 | 0.0227 | 0.0227 | No |

Balanced weights were retained. This is evidence availability and ranking improvement, not a prompt or benchmark-methodology change.

### Candidate-pool ablation

Frozen variables: same conversation, first 50 questions, Nomic, raw representation, identity off, balanced fusion, seed 13, final top-k 5, Qwen 3.5 4B, thinking off. The only changed variable was the pre-reranking candidate pool, 30→60.

| Metric | Pool 30 | Pool 60 | Difference |
|---|---:|---:|---:|
| Candidate evidence recall | 0.7707 | 0.7707 | 0.0000 |
| Fusion evidence recall | 0.7277 | 0.7667 | +0.0390 |
| Final-context evidence recall | 0.5542 | 0.5842 | +0.0300 |
| All-evidence availability | 0.5200 | 0.5200 | 0.0000 |
| F1 | 0.1526 | 0.1800 | +0.0274 |
| BERTScore | 0.1744 | 0.1997 | +0.0253 |
| ROUGE-L | 0.1414 | 0.1682 | +0.0268 |

Category F1 changed from 0.1323→0.1462 for category 1, 0.1763→0.2124 for category 2, and held at 0.0227 for category 3. Pool 60 was retained through the explicit `PRIMA_RETRIEVAL_CANDIDATE_POOL_SIZE=60` production setting.

### Query-expansion ablation

Frozen variables: same conversation, first 50 questions, Nomic, raw representation, identity off, balanced fusion, pool 60, seed 13, final top-k 5, Qwen 3.5 4B, thinking off. The only changed variable was resource-backed query expansion.

| Metric | Expansion on | Expansion off | Difference |
|---|---:|---:|---:|
| Sparse evidence recall | 0.4132 | 0.4732 | +0.0600 |
| Final-context evidence recall | 0.5842 | 0.5842 | 0.0000 |
| F1 | 0.1800 | 0.1833 | +0.0033 |
| BERTScore | 0.1997 | 0.1928 | -0.0069 |
| ROUGE-L | 0.1682 | 0.1758 | +0.0076 |

Category-1 F1 regressed from 0.1462 to 0.1333, category 2 improved from 0.2124 to 0.2225, and category 3 held steady. Because final evidence did not improve and a category regressed, no-expansion was rejected. The temporary switch was removed; production expansion remains enabled.

### Dense contribution

Nomic contributed 0.7617 dense evidence recall to a 0.7707 union candidate recall. Sparse recall was 0.4132. Fusion and final-stage traces prove dense candidates reached final ranking; dense retrieval is active and materially contributes.

## Failure localization

For the retained pool-60 50-question validation:

| Failure class | Count | Percent |
|---|---:|---:|
| A — gold absent from candidate pool | 13 | 26.53% |
| B — gold lost in ranking/final selection | 6 | 12.24% |
| C — complete evidence present, LLM wrong | 0 | 0.00% |
| D — likely evaluation penalty | 2 | 4.08% |
| E — temporal reasoning | 21 | 42.86% |
| F — multi-memory aggregation | 7 | 14.29% |
| G — query understanding/expansion | 0 | 0.00% |
| H — other/ambiguous | 0 | 0.00% |

Every result also retains stored, dense, sparse, fused, reranked, and final evidence recall, so the stored taxonomy can be audited rather than inferred from the final answer alone.

## Graph and reflection

| Component | Result | Decision |
|---|---|---|
| Graph retrieval | **NOT REACHABLE in production QA** | No demonstrated benefit. Do not retain/force a QA graph route without a separate controlled ablation. |
| Reflection | **NOT REACHABLE in production QA** | No demonstrated benefit. Do not report zero activation as usefulness evidence or force activation. |

No graph or reflection change was made.

## Bugs fixed

1. Correct backend/representation/identity fingerprinting and environment-driven canonical pipeline.
2. Backend-dependent memory novelty that discarded almost every Nomic-ingested turn.
3. Lost LoCoMo evidence and source-turn provenance.
4. Missing dense/sparse/fusion/reranker/final traces in QA artifacts.
5. Invalid failure labeling that treated ordinary wrong answers as generation errors.
6. Missing gold-evidence and category-level retrieval metrics.
7. Dataset hash mislabeled as backend fingerprint.
8. `--output-path` not consistently controlling loader, runner, and runtime logs.
9. File log handlers retaining obsolete output roots.
10. Stale `retrieval-v2-debug` reproduction instructions.
11. Dead `candidate_pool_multiplier` configuration key.
12. Candidate-pool size and complete generation settings missing from benchmark metadata.

## Files modified

- `.gitignore`
- `benchmarks/common/runner.py`
- `benchmarks/common/utils.py`
- `benchmarks/locomo/evaluate.py`
- `benchmarks/locomo/experiment.py`
- `benchmarks/locomo/outputs/retrieval_debug_readme.md`
- `benchmarks/locomo/retrieval_validation.py`
- `benchmarks/locomo/runner.py`
- `config/retrieval.yaml`
- `evaluation/results/retrieval_debug_readme.md`
- `evaluation/reports/benchmark_readiness_report.md`
- `evaluation/reports/retrieval_failure_analysis_plan.md`
- `memory/embedding_pipeline.py`
- `memory/maintenance/memory_importance.py`
- `memory/retrieval/retrieval_controller.py`
- `runtime/prima_runtime.py`
- `tests/test_locomo_qa_correctness.py`

## Expected full benchmark outputs

The clean production run creates:

```text
evaluation/qa/
  answers/answers.json
  logs/benchmarks_locomo_loader.log
  logs/benchmarks_locomo_runner.log
  logs/prima_runtime_adapter.log
  logs/run_metadata.json
  metrics/locomo_summary.json
  metrics/metrics.json
  parsed/answers.json
  prompts/prompts.json
  raw/responses.json
  reports/locomo_qa_report.md
```

Before the first improved full run, the handoff command verifies the three frozen baseline hashes and moves the existing `evaluation/qa` directory to `evaluation/qa_baseline_frozen`. The new run then creates the canonical `evaluation/qa` tree. Both paths and diagnostic output roots are ignored by Git. On later reruns, `--clean-output` replaces only the improved `evaluation/qa` output and leaves the frozen archive untouched.

## Remaining blockers

1. The parent source changes are not committed.
2. Both external submodules contain pre-existing user-owned modifications. They were preserved.
3. The 0.70 evidence-availability milestone was not reached on the one-conversation 50-question sample: candidate recall was 0.7707, but final-context recall was 0.5842 and all-evidence availability was 0.52.
4. Full-run QA improvements over the frozen baseline cannot be claimed until the user runs the complete dataset.

These blockers do not prevent execution, but the repository-clean success criterion remains unmet until the worktree policy is resolved.
