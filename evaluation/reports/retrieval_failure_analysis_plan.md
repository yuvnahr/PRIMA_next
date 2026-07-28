# PRIMA-NEXT LoCoMo Retrieval Failure Analysis and Improvement Plan

Date: 2026-07-18  
Branch: `locomo-eval`  
Frozen QA baseline: 10 conversations, 1,986 questions, seed 13, top-k 5, Qwen 3.5 4B, thinking off

## 2026-07-18 controlled-validation update

P0 instrumentation and correctness work is complete and validated without running the full benchmark. Production now records real embedding provenance, gold source IDs, every retrieval stage, category metrics, and evidence-based failure localization.

The first production-parity investigation exposed a backend-dependent admission bug: Nomic admitted only 1/629 turns because raw cosine similarity was used as an absolute novelty score. The shared admission function now uses lexical Jaccard novelty over semantically selected neighbors; Nomic then admitted 597/629 turns and achieved 0.9933 annotated-evidence storage recall on the 50-question controlled sample.

Nomic/raw/identity-off was verified end to end. On the same conversation and 50 questions, changing only fusion weights from sparse-heavy to balanced kept candidate evidence recall at 0.7707, raised final-context evidence recall from 0.3575 to 0.5542, raised all-evidence availability from 0.34 to 0.52, and raised F1 from 0.0911 to 0.1526. No observed category regressed, so the balanced weights were retained.

Increasing only the pre-reranking candidate pool from 30 to 60, with final top-k fixed at 5, raised fusion evidence recall from 0.7277 to 0.7667, final-context evidence recall from 0.5542 to 0.5842, F1 from 0.1526 to 0.1800, BERTScore from 0.1744 to 0.1997, and ROUGE-L from 0.1414 to 0.1682. Categories 1 and 2 improved and category 3 held steady, so pool 60 was retained.

Disabling only resource-backed query expansion raised sparse evidence recall from 0.4132 to 0.4732 but left final evidence recall unchanged at 0.5842, reduced BERTScore from 0.1997 to 0.1928, and regressed category-1 F1 from 0.1462 to 0.1333. No-expansion was rejected and its temporary production switch was removed.

Dense retrieval is proven active: dense evidence recall was 0.7617 versus sparse recall 0.4132, with dense-origin candidates visible through fusion and final selection.

Graph retrieval and reflection remain unreachable in production QA. Neither has demonstrated benefit, so neither was forced or wired in.

The detailed execution evidence and remaining blockers are in `evaluation/reports/benchmark_readiness_report.md`.

## Executive verdict

The QA pipeline is operational, but the retrieval benchmark is not yet scientifically localizable or ready for another full QA run.

The strongest blockers are configuration and observability:

1. Production QA defaults to the `stable` embedding backend because `PRIMA_EMBEDDING_BACKEND` is unset. Nomic is not proven active in the frozen QA baseline.
2. The retrieval campaigns found Nomic + raw representation + identity normalization off to be strongest, while production QA uses stable + semantic representation + identity normalization on.
3. `memory_hits` is answer-token overlap, not gold-evidence recall.
4. Gold evidence IDs and source turn IDs are dropped before QA artifacts are written.
5. Dense, sparse, fusion, and reranker candidate traces exist inside the retrieval controller but are discarded by QA diagnostics.
6. The failure artifact labels almost every wrong answer as `generation_error`, despite zero actual generation errors.
7. Graph retrieval and adaptive reflection are not invoked by the QA answer path, so their zero rates do not measure usefulness.
8. `candidate_pool_multiplier: 12` is present in `config/retrieval.yaml` but is not read by production.

No retrieval logic or benchmark methodology was changed during this audit.

## Frozen baseline integrity

The existing baseline was not overwritten.

| Artifact | SHA-256 |
|---|---|
| `evaluation/qa/raw/responses.json` | `5DCF4626C872398583B1AEEE8239A9A8FBEE5EE94D457E29F2496168FE95453F` |
| `evaluation/qa/answers/answers.json` | `45ADA709824A215D5A81BADD5BD2146C52FF0B17F1B0C4B85E89481EB774239E` |
| `evaluation/qa/metrics/metrics.json` | `011526E878A6E67D0596B48333EB9882390202CAC665B5A4FE37165720F75996` |

Frozen aggregate metrics:

| Metric | Value |
|---|---:|
| Exact Match | 0.212487 |
| F1 | 0.298870 |
| BLEU | 0.124485 |
| ROUGE-L | 0.292536 |
| BERTScore | 0.355429 |
| Reported `memory_hits` | 0.581067 |
| Average retrieved memories | 5.0 |
| Reflection rate | 0.0 |
| Runtime errors | 0 |

The baseline remains the comparison point even though its retrieval provenance is incomplete.

## One-conversation production validation

Only one conversation was executed, as required:

- Conversation: `conv-42`
- Questions: 260
- Provider/model: Ollama 0.32.1 / `qwen3.5:4b`
- Thinking: off; `thinking` was null for all 260 responses
- Completion status: `done_reason=stop` for all 260 responses
- Runtime errors: 0
- Retrieved memories: exactly 5 for all 260 responses
- Exact Match: 0.192308
- F1: 0.274432
- ROUGE-L: 0.266694
- BERTScore: 0.334607
- Reported `memory_hits`: 0.600000
- Mean QA latency: 8,635 ms

This validates pipeline health only. It does not establish a retrieval improvement because it used the current default stable backend.

## What `memory_hits` actually measures

`benchmarks/locomo/evaluate.py::memory_hit` returns 1 when any normalized token from the expected answer occurs anywhere in final retrieved context.

It does not use:

- LoCoMo evidence IDs;
- candidate-pool membership;
- evidence rank;
- complete evidence recall;
- source turn provenance.

It can count generic token overlap as a hit and is not meaningful for adversarial category 5. The `memory_hits >= 0.70` milestone must therefore be redefined as gold-evidence availability, with category 5 reported separately.

Exact text matching between LoCoMo evidence turns and the frozen final contexts gives the following reconstruction for categories 1–4:

- Questions with mappable evidence: 1,531
- Any gold evidence in final top-5: 0.402351
- All gold evidence in final top-5: 0.322665
- Mean gold-evidence recall in final top-5: 0.353278

This is a reconstruction from final context, not candidate-pool recall. Candidate-stage attribution remains unavailable.

## Category audit

Official numeric category meanings are retained: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial.

| Category | N | EM | F1 | ROUGE-L | Current token-overlap `memory_hits` | Any gold evidence in final top-5 | All gold evidence in final top-5 | Mean final evidence recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 multi-hop | 282 | 0.031915 | 0.158858 | 0.148117 | 0.599291 | 0.338078 | 0.035587 | 0.145208 |
| 2 temporal | 321 | 0.018692 | 0.069416 | 0.068620 | 0.404984 | 0.443750 | 0.412500 | 0.426562 |
| 3 open-domain | 96 | 0.114583 | 0.178101 | 0.175603 | 0.562500 | 0.269663 | 0.134831 | 0.182394 |
| 4 single-hop | 841 | 0.205707 | 0.339755 | 0.328988 | 0.845422 | 0.422117 | 0.404281 | 0.413000 |
| 5 adversarial | 446 | 0.500000 | 0.501441 | 0.501441 | 0.201794 | Report separately | Report separately | Report separately |

Multi-hop evidence completeness is the clearest final-context failure: only 3.56% of category-1 questions received all annotated evidence.

Temporal final-evidence availability is materially higher than temporal QA accuracy. This indicates that retrieval is not the only temporal bottleneck; deterministic temporal representation and Qwen reasoning must be separated in later ablations.

## Hypothesis verdicts

### 1. Evidence recall is too low — confirmed, metric definition wrong

The reported 0.5811 is not evidence recall. Reconstructed final-context recall for categories 1–4 is about 0.3533. Exact candidate-pool and ranking-stage recall cannot be recovered from the frozen QA artifact.

### 2. Dense retrieval may not contribute correctly — confirmed configuration gap

- `PRIMA_EMBEDDING_BACKEND` is unset.
- `memory/embedding_backend.py` defaults to `stable`.
- The production command did not select Nomic.
- QA provenance records the dataset SHA under `backend_fingerprint`, not the embedding pipeline fingerprint.
- QA artifacts contain no dense candidate stage.
- Production fusion weights are dense 0.15, sparse 0.65, temporal 0.10, graph 0.10.
- The active reranker is the lexical fallback because the cross-encoder is disabled.

The completed dense-only campaigns show that Nomic is promising:

| Configuration | Recall@5 | Recall@10 | Candidate success | MRR |
|---|---:|---:|---:|---:|
| Stable + semantic + identity on, one conversation | 0.010152 | 0.015228 | 0.071066 | 0.010081 |
| Nomic + semantic + identity on, one conversation | 0.329103 | 0.419205 | 0.695431 | 0.254356 |
| Nomic + raw + identity off, full retrieval campaign | 0.409464 | 0.515587 | 0.734209 | 0.326786 |

These campaigns used dense-only retrieval over every ingested turn. They do not prove Nomic contribution inside production hybrid QA.

### 3. Query expansion causes drift — confirmed examples, causal effect unmeasured

The frozen artifacts contain generic expansion terms at high frequency:

- `city`: 547
- `country`: 547
- `occupation`: 546
- `identity`: 546
- `profile`: 546

For “What kind of interests do Joanna and Nate share?”, the expansion was `identity profile occupation city country`. This is unrelated to shared interests.

Expansion affects the sparse lexical query; dense embedding uses the original query. A current-vs-no-expansion retrieval ablation is required before changing behavior.

### 4. Correct evidence is found but ranked poorly — partially confirmed

Final contexts show cases where decisive evidence is below generic evidence, but the frozen QA artifact omits dense, sparse, fusion, and reranker ranks. Exact candidate-to-final losses cannot be counted.

### 5. Final top-k 5 may be restrictive — plausible, not yet isolated

Multi-hop completeness is very low, but increasing final context could add noise. Production already generates at least 30 candidates, while `candidate_pool_multiplier: 12` is ignored. Candidate depth and final context size must be tested separately.

### 6. Temporal reasoning is weak — confirmed end-to-end, source split unknown

Category-2 EM is 0.0187 and F1 is 0.0694 while all annotated evidence reaches final top-5 for 41.25% of temporal questions. The temporal strategy scores recency relative to current wall-clock time, not the conversation/question reference time. A deterministic temporal-normalization ablation is justified only after retrieval-stage localization.

### 7. Multi-memory aggregation is weak — confirmed

Category-1 all-evidence availability is 0.0356 and mean evidence recall is 0.1452. Independent top-5 turns rarely provide complete multi-hop evidence. Evidence grouping is a P2 experiment, after candidate recall and ranking are fixed.

### 8. Identity normalization may hurt — confirmed by completed ablation

For Nomic + raw on the full retrieval campaign:

- Identity off Recall@5: 0.409464
- Identity on Recall@5: 0.392039

Production remains identity-on. Identity-off should be evaluated as one explicit configuration change, not bundled with other retrieval changes.

### 9. Graph retrieval is unused — confirmed unreachable in QA

The production `RetrievalController` strategies are dense, sparse, and temporal. `GraphTraversalStrategy` is only reachable through a separate router that QA does not instantiate. `graph_links_traversed` merely serializes links already attached to selected notes; it is not traversal evidence.

### 10. Reflection is never used — confirmed unreachable in QA

`answer_question()` directly retrieves context and calls Qwen. It does not invoke `ReflectionEngine`. The reported reflection flag only detects selected memories carrying reflection metadata. Zero reflection rate therefore does not test reflection thresholds or utility.

### 11. LLM and retrieval failures are conflated — confirmed

The frozen run has zero actual generation errors, but its 1,564 failure records contain:

- `generation_error`: 1,562
- `entity_resolution_failure`: 2

`failure_record()` assigns `generation_error` to a wrong answer whenever `llm_used` is true. This taxonomy is invalid.

### 12. Aggregate metrics hide category behavior — confirmed

Category EM ranges from 0.0187 to 0.5000, and final evidence completeness ranges from 0.0356 to 0.4125 for categories 1–4. Aggregate-only acceptance would hide severe multi-hop and temporal weaknesses.

## Additional architecture inconsistencies

### Campaign ingestion differs from production ingestion

`evaluation/experiments/run_campaigns.py` stores every turn using deterministic IDs such as `conversation:evidence_id`.

Production QA:

- replays turns through the full workflow;
- applies `MemoryImportanceEngine` with threshold 0.55;
- may discard turns;
- creates UUID memory IDs;
- does not retain source evidence IDs in stored note context.

Retrieval campaign gains therefore do not automatically transfer to production QA.

### Campaign fingerprints do not uniquely identify representation settings

`current_embedding_metadata()` hardcodes semantic representation and identity-on. Raw and identity-off campaign rows can share the same recorded fingerprint. Stored-note metadata is more specific, but the campaign-level fingerprint is not sufficient for reproduction.

### Configured candidate depth is ignored

`config/retrieval.yaml` contains `candidate_pool_multiplier: 12`. `HybridFusionConfig.from_file()` reads only weights, and `RetrievalController` keeps its constructor default multiplier of 6. With top-k 5, production tracks 30 candidates instead of the configured 60.

## Required failure localization contract

Every QA result must retain compact, deterministic evidence provenance:

1. Question ID and official category.
2. Gold evidence source IDs.
3. Created/stored memory source IDs and admission decision.
4. Dense top-N IDs, raw similarities, ranks, and embedding fingerprint.
5. Sparse top-N IDs, scores, and ranks.
6. Fused top-N IDs, component scores, and ranks.
7. Reranked top-N IDs, scores, and ranks.
8. Final context IDs and evidence ranks.
9. Qwen raw response, parsed answer, and generation status.
10. Per-question metrics.

Failure classes:

- A: gold memory never created/admitted;
- B: gold memory stored but absent from dense and sparse candidate pools;
- C: gold memory generated by a source strategy but lost in fusion;
- D: gold memory survives fusion but is lost in reranking/final selection;
- E: complete gold evidence reaches final context but Qwen answers incorrectly;
- F: answer is semantically acceptable but penalized by evaluation;
- G: temporal reasoning failure;
- H: multi-memory aggregation failure;
- I: query understanding/expansion failure;
- J: ambiguous or unclassified.

The user-requested A–H presentation can combine C/D as ranking loss and G/H/I under “other” when required, but the stored artifact should preserve the more precise stage.

## Prioritized roadmap

### P0 — correctness and observability

These changes do not alter retrieval ranking:

1. Record dataset SHA separately from the actual embedding backend fingerprint.
2. Record backend, model, dimensions, representation mode, identity mode, and all retrieval weights.
3. Preserve LoCoMo evidence IDs in `BenchmarkResult`.
4. Attach source conversation/session/turn IDs to created memories.
5. Write compact stage candidate IDs, component scores, and ranks from existing controller diagnostics.
6. Replace token-overlap `memory_hits` with:
   - candidate evidence recall;
   - final-context evidence recall;
   - all-evidence availability;
   - mean evidence rank.
   Keep the old value under a clearly named legacy field for baseline comparison.
7. Replace the current failure classifier with evidence-stage localization.
8. Emit official category breakdowns and a category regression table.
9. Add an explicit one-conversation guard to every diagnostic CLI.
10. Validate that production configuration exactly identifies the intended Nomic/raw/identity setting before any full QA run.

Files/components:

- `benchmarks/common/runner.py`
- `benchmarks/common/runtime_adapter.py`
- `benchmarks/locomo/evaluate.py`
- `benchmarks/locomo/experiment.py`
- `runtime/prima_runtime.py`
- `memory/embedding_pipeline.py`
- `memory/memory_note.py`

Validation:

- focused unit contract;
- compile/lint;
- one-conversation retrieval trace;
- one-conversation QA run;
- no full QA run.

Rollback criterion: any change alters retrieved ordering, prompts, or frozen metric computation.

### P1 — highest-impact retrieval ablations

Run each change alone on the same seeded conversation before wider evaluation:

1. **Production parity:** current production versus explicit Nomic + raw + identity-off, with identical ingestion and hybrid retrieval.
2. **Dense contribution:** full hybrid versus identical hybrid with dense removed. Report candidate and final evidence deltas by category.
3. **Query expansion:** current expansion versus disabled expansion.
4. **Candidate depth:** 30 versus 60 candidates, final top-k fixed at 5.
5. **Fusion/reranking localization:** no weight change until stage traces show where gold evidence is lost.
6. **Identity:** identity-off versus identity-on as its own ablation if not already isolated by production parity.

Acceptance:

- primary: gold final-context evidence recall reaches at least 0.70 as the first milestone, stretch 0.80;
- no material category regression;
- gain is attributable to one changed component;
- improvement survives at least one held-out conversation before a full run;
- rollback if aggregate gain is caused by a major temporal or multi-hop regression.

### P2 — reasoning and evidence improvements

Only after P1 localization:

1. deterministic temporal normalization using memory timestamps;
2. coherent grouping of multi-memory evidence;
3. query-aware evidence scoring using general entity, event, relation, and temporal agreement;
4. adaptive final context size while keeping candidate depth fixed.

Each is a separate ablation. Do not combine temporal normalization with aggregation or ranking changes.

### P3 — optional research

1. **Reflection:** wire a controlled second pass only for legitimate low-confidence, conflict, or incomplete-evidence cases. Retain only if it improves evidence availability or QA without category regression.
2. **Graph:** build and query a graph only for relational/multi-hop routes. Retain only if graph candidates add unique gold evidence and improve downstream QA.
3. cross-encoder reranking;
4. alternative event representations.

Graph/reflection utilization alone is not a success metric.

## Ablation matrix

| Step | Frozen variables | Single changed variable | Primary measure | Secondary measure |
|---|---|---|---|---|
| A0 | Current production | P0 instrumentation only | identical answers/retrieval | complete trace coverage |
| A1 | A0 | explicit Nomic/raw/identity-off production parity | candidate/final evidence recall | category QA metrics |
| A2 | A1 | remove dense | dense marginal contribution | rank changes |
| A3 | best prior | disable expansion | candidate/final evidence recall | drift cases |
| A4 | best prior | candidate depth 30→60 | candidate recall | final top-5 unchanged |
| A5 | best prior | one reranking change | final evidence rank | QA metrics |
| A6 | best prior | temporal normalization | temporal QA | non-temporal regression |
| A7 | best prior | evidence aggregation | multi-hop QA | context noise |
| A8 | best prior | reflection second pass | low-confidence subset | latency/cost |
| A9 | best prior | graph route | relational subset | unique evidence gain |

## End-to-end acceptance gates

Primary milestone:

- gold-evidence availability at final context >= 0.70;
- stretch >= 0.80;
- exact localization shows which stage produced the gain.

Secondary full-run milestones, using the frozen evaluator for comparability:

- EM >= 0.27;
- F1 >= 0.36;
- BERTScore >= 0.42;
- ROUGE-L must improve over 0.2925.

These are evaluation gates, not optimization targets.

Non-numeric gates:

- Nomic contribution proven end-to-end;
- every failure assigned to an evidence or generation stage;
- no material category regression;
- one change per ablation;
- no LoCoMo-specific rules or answer tuning.

## Recommended sequence

Frozen baseline
→ add provenance and failure localization without changing retrieval
→ validate on one conversation
→ explicit production-parity Nomic ablation
→ measure dense contribution
→ expansion ablation
→ candidate-depth ablation
→ ranking ablation only if localization supports it
→ temporal and multi-memory ablations
→ graph/reflection only if reachable and useful
→ full LoCoMo QA rerun only after a configuration survives controlled validation

## Current blockers to a full benchmark

1. Nomic is not selected or fingerprinted in production QA.
2. Production ingestion is not equivalent to the completed retrieval campaigns.
3. Gold evidence provenance and candidate-stage traces are absent.
4. `memory_hits` and failure taxonomy are invalid for scientific attribution.
5. Category-level regression reporting is missing.
6. Graph/reflection usefulness cannot be evaluated because QA does not invoke them.
7. The worktree contains pre-existing modified submodules:
   - `benchmarks/locomo/external`
   - `benchmarks/locomo-v2/external`

Do not run the full benchmark until P0 is complete and the intended production retrieval configuration passes one-conversation validation.
