# Person 1 LoCoMo/runtime remediation handoff

## Checkpoint and provenance

Goal 1 (A1): complete at `728709be17f7f8b08f67d01fb06e07c160bd612b`. Goal 2 (A2): complete at `6e6f968aa9c219b0ffe86d4d3fa2e4d534830f39`. Goal 3 (A3): implementation and deterministic acceptance complete; its checkpoint SHA is reported in the completion message. Goal 4 has not started. Each goal's ledger records its own checks and limitations.
Actual base: `c25a178d6308bc48b3a5e44d9519316a5179c499` (the planning snapshot); no intervening tracked implementation changes.
Predecessor: none. The completed checkpoint SHA is reported in the completion message, not inside its own commit.

Delivery checkout: `C:/PRIMA_next/tmp/person1-dev-p1`, branch `dev-p1`.
This is a clean local clone of the assigned branch because the original `C:/PRIMA_next` checkout contained unrelated deletions/untracked files and a dirty LoCoMo-v2 submodule. Those files, its index and branch pointer were left alone. No push, PR, merge or production execution is authorized by this checkpoint.
The interpreter is `C:/PRIMA_next/.venv/Scripts/python.exe`, Python 3.14.8. Sandbox execution could not access its base installation; approved execution outside that boundary uses the same existing environment, without dependency changes.

Scope source: `Person_1_LoCoMo_Runtime_Implementation.md`, SHA-256 `43bf76b274adba04947e630eef9fea767d493c2b4976a72090656b4b7f914dab`.
The first attachment invoked Goal 1; `1ae70693-1579-4147-9fbc-5d1d0d20f96e/pasted-text-1.txt` invoked Goal 2; the latest `3674c88a-9fbf-4d05-9518-a3fa63f0a7ac/pasted-text-1.txt` invokes Goal 3 only. The newer four-goal operating contract supersedes the source plan's intermediate PR/merge workflow.
The LoCoMo external checkout was initialized from the existing local pinned repository, without a download or external-source edit. Other external submodules remain uninitialized in this delivery clone.

## Requirement coverage and affected callers

| Requirement | Implementation | Regression evidence |
|---|---|---|
| Capture each retrieval response before replacement, without full tracing | `reasoning/controller.py`, defaulted `EvidenceState.retrieval_hops`, `reasoning/evidence_audit.py` | Earlier-hop gold followed by last-hop miss; unknown stage availability |
| Preserve outer retries and accepted evidence before compression | `workflow/prima_workflow.py`, typed request-owned lists in `workflow/execution_context.py` | Actual canonical PRIMA_FULL retry, phase-exception retry, global hop indices and attempt identity |
| Report configured stage limits | `memory/retrieval/retrieval_controller.py` adds diagnostic limits only | Existing retrieval caller suite plus audited synthetic limits |
| Audit actual serialized evidence and valid model citations separately | `workflow/answer_generation.py`, `runtime/prima_runtime.py` | Actual fake-provider prompt payload after compression; unknown labels rejected; fallback is not a citation |
| Deduplicate turns, retain memory identity and count untraceable records | `reasoning/evidence_audit.py`, runtime output shaping | Multiple memories for one turn, separate conversations, untraceable summary, excluded memory |
| Separate correctness from evidence and preserve deterministic failure precedence | `benchmarks/locomo/evaluate.py` shared classifier; experiment compatibility accessor | Correct answer with annotation miss; admission/retrieval/selection precedence; partial coverage; F1=1 is not EM |
| Category-5 and unannotated denominators | Evaluator evidence populations and structured abstention metric | Annotated adversarial abstention excluded from answerable recall; no-annotation means null |
| Official and legacy scores coexist | `benchmarks/locomo/official_scoring.py`, evaluator and per-question scores | Ten category/multi-answer/date/refusal equivalence cases against the frozen pinned upstream core |
| Preserve raw reference metadata | `benchmarks/locomo/adapter.py` | Category-3 semicolon reference and adversarial metadata survive adaptation |
| Keep old records readable and new checkpoints resumable | Experiment records/checkpoints, evaluator, manifest identities | Old/new JSON contract fixture; checkpoint roundtrip; existing interrupted/resumed experiment test |
| Avoid optional heavyweight metrics in core evaluation | Isolated scorer imports no BERTScore | Separate-process core evaluation import check |

Call searches covered `ReasoningController.answer`, `EvidenceState`, `ExecutionContext`, `AnswerResult` diagnostics, `EvidenceAcquisitionController`, generation `_prompt`/`_parse`/`_fallback`, runtime output and compatibility adapters, `_retrieval_audit`, `_question_record`, `_classify_failure`, evaluator helpers and existing caller tests. No public constructor argument was renamed or removed. `EvidenceState.retrieval_hops` and `ExecutionContext.retrieval_hops`/`prompt_evidence` default to empty lists; `GenerationResult.failure_kind` defaults to null. `ReasoningController.answer` accepts optional keyword-only `retrieval_hops=None` and `attempt_index=0`, allowing the workflow to retain observations even when integration raises before returning. These observations are typed request state, never caller-supplied metadata. No RuntimeConfig fields or production policies changed in Goal 1.

## Producer/consumer contract for Person 2

Runtime response envelope stays version 1.0. New `output_data.evidence_audit` is version **1.0**, scope `request_all_hops`, available in standard diagnostics. It includes:

- `hops`: globally numbered `hop_index`; `attempt_index` for workflow acquisitions; `attempt_kind` (`initial`, `follow_up`, `workflow_retry`); actual retrieval query; per-stage source-turn and raw-memory IDs; configured limits; accepted/excluded IDs; retrieval-only elapsed milliseconds; terminal stop reason on each acquisition's last hop.
- Stages: `dense`, `sparse`, `fused`, `reranked`, `returned`. Full existing candidate diagnostics are used; the audit is not artificially limited to the legacy top-30 views. Missing stage diagnostics are null, never inferred from the returned set.
- `stage_source_turn_ids`: all-hop stage unions. `candidate_source_turn_ids` is the union across observed stages/hops, or null when any required stage observation is unavailable. The per-hop observations remain available in that case.
- `accumulated_source_turn_ids` / `accumulated_memory_ids`: evidence accepted by the integrator, before prompt compression, across acquisitions. This records acceptance, not final prompt exposure.
- `supplied_source_turn_ids` / `supplied_memory_ids`: evidence records actually serialized in the **last model prompt**, after the existing compressor. This may be empty when no generation occurred. It does not claim that a provider attended to every supplied token.
- `cited_source_turn_ids` / `cited_memory_ids`: valid final model-selected references mapped through that prompt. Abstentions, suppressed final answers and extractive/error fallbacks do not create model citations.
- Separate untraceable accumulated/supplied/cited memory counts. Hops carry untraceable accepted IDs; there is no gold-turn recall credited to untraceable summaries.

Source-turn identity is `conversation_id:source_turn_id`, e.g. `conv:D1:1`; raw memory IDs never share that namespace. With no conversation ID on a standalone reasoning request, source turns remain local IDs. Runtime LoCoMo exports always have the conversation ID. No repeated evidence text is stored in this compact audit.

Existing `output_data.retrieval.stage_source_ids` is unchanged and explicitly labeled `scope=last_retrieval_call`. It is not an all-hop union. `AnswerResult.answer_diagnostics.evidence_audit` exposes the acquisition audit (including earlier attempts when a workflow collector is supplied); runtime `output_data.evidence_audit` adds final prompt/citation fields. Diagnostic `retrieval_count`/`retrieval_call_count` now count all returned retrieval responses in the request, including outer/phase retries, rather than the final reasoning invocation alone. Calls that throw before returning a response are not included, preserving the prior successful-return convention. `retrieval_result_count` remains the last response's result count.

Question records and aggregate metrics are version **2.0**. Question records export the four qualified unions, the compact runtime audit and named `answer_scores`. Existing `candidate_evidence_ids` remains conversation-local but is now sourced from the all-hop audit for schema 2.0. It is null when unavailable. Schema-1 candidate fields represented last-call data and must never be treated as all-hop recall.

Compatibility aliases retained:

| Old key | Explicit meaning |
|---|---|
| `f1` | `legacy_squad_f1`, token overlap using existing SQuAD normalization |
| `bleu` | `smoothed_sentence_bleu`, existing NLTK method-1 smoothed sentence BLEU |
| `final_evidence_ids`, `final_evidence_recall` | Legacy selected/cited-source view; schema 2.0 excludes error/extractive fallbacks |
| `average_retrieved_evidence` | Deprecated alias of `average_cited_evidence`; never a retrieval-candidate count |
| `abstention_category_5_accuracy` | Structured-outcome accuracy over records with an observed outcome; missing outcomes are unavailable |

`official_locomo_f1`, answerable official F1, per-category official scores, `official_adversarial_raw_text_score` and scorer identity are separate from all legacy metrics. Refusal text is not rewritten: an existing typed abstention can score 1 under the structured metric and 0 under the official text rule. Strict normalized EM remains order/multiplicity sensitive, even when token F1 is 1.

The declared correctness rule remains strict legacy normalized EM for answers, plus successful structured category-5 abstention. Execution/ingestion/parse errors take priority. Only incorrect annotated answers receive primary admission → all-hop retrieval → supplied-selection → generation/scoring diagnostics. Partial supplied gold coverage is incomplete coverage and does not establish full multi-hop support. Independent incomplete-evidence flags can overlap and can appear on correct answers; they are annotation diagnostics, not causal proof.

`evidence_populations.answerable` and `.adversarial` report annotated counts, per-field observed denominators, recall and non-additive flag counts. Missing annotations are `not_applicable`; missing observations are null/unavailable; observed empty evidence is a real zero. Old last-call data is excluded from new all-hop metrics. Evaluator recalculates primary classifications in memory, without overwriting archived predictions.

`GenerationResult.failure_kind` differentiates provider failure from invalid response parsing, even when fallback produces an abstention. Such fallback records do not count as successful structured abstentions.

## Official scorer identity and license

- Upstream: `snap-research/LoCoMo`, pinned submodule commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`.
- Scorer path: `task_eval/evaluation.py`, Git blob `8f597dd687e66832da1f6f04a169e05622049576`.
- Source SHA-256 after LF newline normalization: `8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd`.
- License and attribution retained in `benchmarks/locomo/official_scoring_LICENSE.txt` and the adapted module. Core stemming, comma-separated category-1 matching, category-3 first-semicolon reference and category-5 literal refusal handling are preserved; optional metrics/file I/O are omitted.
- The adapted file's actual SHA-256 is emitted with each scorer identity. The offline upstream core fixture (trailing whitespace normalized) has normalized-text SHA-256 `fba8820873e0d0194e50d36c2a774e1020bed4134bc0170373c36b5fbecfd4e5`. Tests compare function ASTs to the verified pinned source when initialized; fixture identity/equivalence tests also run offline without the submodule.

## Artifacts, fingerprints and resume

Small contract examples and expected consumer outputs: `tests/fixtures/locomo_review/report_contract.json` (one old, one new); the new fixture intentionally has accumulated recall 1 and supplied recall 0. It was generated from the actual canonical runtime with synthetic retrieval and a fake provider, then compacted by omitting the duplicate raw response and normalizing timing to zero. Its audit is internally consistent with the serialized prompt.
Existing checkpoint serialization preserves the new audit unchanged. No historical prediction is rewritten. New manifests add evidence/question schema versions and the exact scorer identity to `benchmark_config`; incompatible old manifests cannot silently resume into the new scorer/schema. Old checkpoint records remain readable for explicit offline evaluation, with unavailable new diagnostics. Person 2 must consume this contract rather than reconstruct supplied/cited context from retrieval or accumulated IDs.

## Validation ledger

All commands below use the supplied environment from the delivery checkout.

- Initial focused suite: 81 passed / 3 expected old-semantics assertions failed. Updated two failure fixtures to use incorrect answers (a correct answer can no longer be a retrieval failure), and the fake runtime now supplies explicit new-schema observations.
- Initial new fixtures exposed a missing `string` import in the frozen upstream setup and an incorrect expected last-call fake result. Fixed both test issues without changing scorer behavior.
- Affected caller suite: **233 passed in 68.33 seconds** across LoCoMo, prompt identity, reasoning, workflow, runtime, retrieval, memory, lineage and evaluation tests, before the final added fallback/import regressions.
- Expanded affected suite: **251 passed in 64.84 seconds**; final focused recheck before the phase-exception hardening: **97 passed in 22.55 seconds**. Strict mypy initially found one unannotated list; the corrected 31-source run passed. The subsequent phase-exception review moved audit collection into typed request state and adds two further regression tests.
- Final expanded suite after phase-exception hardening: **255 passed in 65.36 seconds**. After the final runtime counter correction, the new regressions and runtime principal-boundary checks passed again: **45 passed in 18.73 seconds**.
- Staging revealed whitespace-only issues in the preserved upstream fixture/license; normalized trailing whitespace, refreshed the fixture hash and verified AST equivalence again. The final LoCoMo/new-regression check after preferring authoritative citation IDs for cited averages: **53 passed in 21.75 seconds**; Ruff and staged/unstaged diff checks passed again.
- Owned-path Ruff: **all checks passed**. Strict affected-package mypy: **success, no issues in 31 source files** (only existing unused override-section notes). `git diff --check`: **passed** from the delivery checkout; the initialized pinned external submodule has no diff.
- An initial diff command in an elevated combined shell could not identify the Git checkout; the normal sandbox Git check was rerun successfully. No Git configuration was weakened. A one-off fixture generator wrote its output but hit a Windows open-log-handle error during temporary cleanup; the generated fixture independently passes the final JSON/consumer checks.

Final validation commands (working directory `C:/PRIMA_next/tmp/person1-dev-p1`):

```powershell
C:/PRIMA_next/.venv/Scripts/python.exe -m pytest -q tests/test_locomo_review_regressions.py tests/test_locomo_qa_correctness.py tests/test_answer_prompt_identity.py tests/test_embedding_backend.py tests/reasoning tests/retrieval tests/memory tests/evolution tests/lineage tests/workflow tests/reflection tests/runtime tests/evaluation
C:/PRIMA_next/.venv/Scripts/python.exe -m pytest -q tests/test_locomo_review_regressions.py tests/runtime/test_principal_hardening.py
C:/PRIMA_next/.venv/Scripts/python.exe -m pytest -q tests/test_locomo_review_regressions.py tests/test_locomo_qa_correctness.py
C:/PRIMA_next/.venv/Scripts/python.exe -m ruff check benchmarks/locomo memory reasoning workflow uncertainty reflection runtime config/runtime_config.py tests/test_locomo_qa_correctness.py tests/test_locomo_review_regressions.py tests/fixtures/locomo_review
C:/PRIMA_next/.venv/Scripts/python.exe -m mypy reasoning runtime workflow memory/retrieval/retrieval_controller.py
git diff --check
git -C benchmarks/locomo/external diff --exit-code
```

No exact historical prediction archive was supplied or located in `C:/PRIMA_next/benchmarks/locomo/outputs` (only log files were found). No historical score reproduction, fresh model inference, model download, GPU experiment or campaign pilot was performed. Synthetic correctness is not evidence of a benchmark-quality gain. Full CI-equivalent checks and actual Kaggle validation belong to the final checkpoint; Goal 1 uses affected suites and lint/diff checks.

## Remaining sequence and rollback

Goal 2 requires this committed checkpoint and tests, then historical provenance/captions and native learned retrieval with typed identities. Goal 3 repairs query drift and both answer gates/budgets. Goal 4 handles evidence eligibility/admission/reflection and the final broad review package. None is implemented or promoted by Goal 1.

Person 2 can implement old/new contract consumers using the committed fixtures. Final actual producer/consumer integration waits for Person 1 Goal 4, in a separate integration checkout at pinned parent/candidate SHAs. Production remains disabled until human-reviewed integration and fresh measured identity/resource/time gates; full LoCoMo acceptance remains all 10 conversations / 1,986 questions exactly once.

Rollback: scoped revert of this checkpoint, preserving evidence artifacts. Use a new run identity after a schema/scorer change; never resume incompatible checkpoints or reset `dev`. No storage migration, embedding default, answer-gate policy or reflection threshold was changed.

## Goal 2 — provenance and native semantic retrieval

Implementation predecessor/base: `728709be17f7f8b08f67d01fb06e07c160bd612b`, verified as HEAD and ancestor before editing. There were no intervening changes or staged work. The same isolated `dev-p1` clone is used; the original checkout and Person 2's files remain untouched.

### Requirement coverage

| Source requirement | Producer/config changes | Evidence |
|---|---|---|
| A2.1 historical timestamps and numbered sessions | Adapter preserves normalized/original time, precision and speaker; workflow validates replay time before any state/admission work and passes `timestamp=` to `MemoryNote.create` in normal and controlled ingestion | All three ingestion policies; malformed/empty/missing/non-string time failures; live-time default; record and metadata-codec reload |
| Conversation identity separate from source session | `source_conversation_id` is forced from request isolation; `source_session_id` retains the numbered session; retrieval requests exclude explicitly foreign conversation sources before fusion | Two conversations sharing a repository; existing integration replay/QA test |
| Source-relative temporal signals | `RetrievalRequest.reference_time/historical` and corresponding query analysis fields; historical unanchored queries use neutral time signals, not today's clock | Source-year vs execution-year check; old histories are not aged out using the execution date |
| A2.2 caption representation | Explicit `include_captions=False`; only the pinned dataset's supplied `blip_caption` is appended with `Image caption:` | Caption switch/fingerprint tests; original dataset hash remains unchanged; answer/gold changes cannot affect replay representation |
| A2.3 native learned dimensions and model identities | Existing backend no longer resizes learned vectors; explicit device/revision/local snapshot/batch/thread settings; staged-only loading, no remote code; exact artifact hashes | Wrong dimensions/nonfinite/zero vectors rejected; cached pinned MiniLM CPU execution and held-out paraphrase ranking measured |
| Caches and stale state | Bounded loaded-model caches keyed by identity/revision/device/snapshot/thread limit; vector metadata includes learned identity; in-memory retrieval rejects incompatible vectors/fingerprints; explicit `rebuild_embeddings` preserves content/time/context | Revision/device separation, stale fingerprint/dimension rejection and explicit rebuild tests; incompatible experiment resume rejected |
| A2.4 fusion/reranker and costs | Typed strategy/weight overrides; existing cross-encoder gains pinned CPU/local-only loading and bounded batches; invalid counts/nonfinite scores fail; stable source-ID ties; stage timing diagnostics | Empty/constant/tie tests; fake cross-encoder cache and CPU-option checks; frozen real-data CPU ladder |
| Person 2 staging/configuration and compatibility | Runtime construction/reset, embedding context, retrieval calls, manifest, LoCoMo experiment, CLI and resume identity all consume the typed settings | Roundtrip/override tests, affected caller checks, old/new Goal 1 audit/scorer regressions |

Caller searches covered all `EmbeddingBackendConfig`, `embed_text`/`stable_embedding`, model loader/cache, `current_embedding_metadata`, repository query/rebuild, `RetrievalRequest`, temporal agreement, fusion/reranker, `MemoryNote.create`/reload, ingestion/replay controllers, runtime construction/reset and LoCoMo experiment/CLI users. Existing benchmark/preflight/notebook/dependency/CI files owned by Person 2 were not edited. The shared integration replay fixture continues to work through its explicit `[ISO date]` transcript prefix; missing metadata never substitutes the execution clock.

### Frozen typed configuration and producer contract

Defaults retain stable-hash 64-dimensional embeddings, semantic representation, identity normalization and lexical reranking. Learned native dimension is required explicitly in candidate configurations (384 for MiniLM); a wrong 64-dimensional learned configuration fails instead of changing a learned vector. Environment-based learned selection defaults to 384, while ordinary stable selection stays 64. No answer gate, generation budget, admission threshold or reflection policy changed.

| Added `RuntimeConfig` field | Default |
|---|---|
| `embedding_revision` | null; a learned loader requires an immutable 40-hex revision |
| `embedding_device` | `cpu` |
| `embedding_local_snapshot` | null: resolve the pinned revision from the existing HF cache only |
| `embedding_batch_size` | 16 |
| `retrieval_cpu_threads` | 2 (process-wide PyTorch CPU thread bound when loading an auxiliary model) |
| `retrieval_strategies` | null: retain existing profile routing |
| `retrieval_weights` | null: retain existing `config/retrieval.yaml` weights |
| `reranker_revision` | null; required for actual learned loading |
| `reranker_device` | `cpu` |
| `reranker_local_snapshot` | null: pinned local HF cache only |
| `reranker_batch_size` | 8 |

`EmbeddingBackendConfig.from_runtime_config` is the shared conversion used by runtime and experiments. Context overrides retain all fields when only dimension changes. Runtime repository preflight now runs inside the effective embedding/pipeline context, so persistent fingerprint checks see typed representation settings. Runtime reset reconstructs retrieval using the same typed controls. `runtime_manifest()` records actual loaded embedding identity/device/artifact hashes, reranker diagnostics and effective fusion weights.

Both learned loaders always set `local_files_only=True`, `trust_remote_code=False` and the explicit device. There is no constructor fallback that drops these settings. Legacy `nomic` selection cannot enable remote code and will fail if its artifact requires that capability. Device `auto` is rejected by these auxiliary constructors. CPU thread settings are process-wide; a future overlapping workload must use a compatible bound and fresh measurements, not assume per-model independent CPU budgets.

Stage diagnostics add `timings_ms` for query embedding, dense lookup, sparse lookup, fusion and reranking. Compact Goal 1 hops carry this additive field. Query embedding is computed once per retrieval call, rather than once per memory type. Source-qualified candidate/accumulated/supplied/cited fields, audit version 1.0 and question/metric version 2.0 retain their Goal 1 meanings.

Normal/controlled replay notes preserve `source_timestamp`, `source_timestamp_original`, `timestamp_precision`, `source_timestamp_source`, numbered `source_session_id`, `source_turn_id`, `speaker` and `source_conversation_id`. Legacy callers with no timestamp metadata may use an explicit leading ISO transcript timestamp; malformed supplied metadata fails, and missing numbered sessions remain unknown. Ordinary live ingestion retains current time. Explicitly conversation-tagged memories cannot cross request conversations; legacy unscoped documents retain their existing shared behavior. In production LoCoMo each history still has its own runtime/repository.

Historical QA requests set `historical_context=true`. Optional ISO `reference_time` is an explicit source/question anchor; there is no fabricated last-session anchor when a question provides none. Temporal ranking uses source-session timestamps conservatively and does not claim to resolve every event date embedded in prose. Caption inclusion is an input ablation, with no image downloads, vision inference or generated captions.

### Model staging and exact identities

Measured embedding candidate: `sentence-transformers/all-MiniLM-L6-v2`, revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, CPU, native 384 dimensions, batch 16, two CPU threads. Actual artifact-set SHA-256: `7550e754abbd9086d0f997f2fda4ff22744ac6815bea4550123b73e40b528a7e`. Model files already existed locally; nothing was downloaded.

The optional cross-encoder candidate explicitly selects the canonical repository spelling `cross-encoder/ms-marco-MiniLM-L6-v2`, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, CPU, batch 8, two threads. This is separate from the unchanged legacy default string with `L-6`. The canonical repository/history was verified through [its official history](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/commits/main) and [model metadata](https://huggingface.co/api/models/cross-encoder/ms-marco-MiniLM-L6-v2). Cross-encoder files are not cached locally; actual execution remains explicitly pending, and lexical output is never labeled cross-encoder output.

Person 2 must stage the exact revisions before production, including configuration, tokenizer, pooling (for MiniLM), and weight files. A configured custom `*_local_snapshot` directory requires `prima_model_manifest.json` with `model_identifier`, `revision` and a complete `files` mapping of relative paths to SHA-256 values. The loader enumerates actual non-hidden files (excluding the manifest itself), compares the entire map and rejects identity/hash mismatch; it never trusts manifest paths as file-operation targets. For a pinned HF cache, the loader resolves the exact revision locally and hashes the artifacts actually supplied to the model. Full model/file identities are stored once in runtime/experiment manifests, while each learned note stores the aggregate snapshot hash.

Model cache limit is four entries per loader; keys include revision, explicit device, local snapshot and CPU thread setting. No new general vector cache was added. Existing in-memory normalized-vector caching invalidates on vector or fingerprint changes. Explicit in-memory rebuild computes replacement vectors before applying them and retains historical fields. No storage migration or persistent collection-dimension migration is performed by this checkpoint.

Learned embedding fingerprints include model, native dimension, revision, device, snapshot artifact hash, L2-float32 normalization, representation/identity version, batch and thread settings. Stable-hash fingerprints retain their existing values. LoCoMo manifests additionally freeze full typed runtime settings, auxiliary artifact identities, `retrieval_contract_version=2.0` and `locomo-replay-v2` input representation identity/caption policy. Changed model/revision/device/dimension/representation/policy/configuration invalidates resume; old artifacts remain readable rather than silently mixed into a new run.

### Development controls and measurements

Committed controls: `tests/fixtures/locomo_review/goal2_development_manifest.json` and `goal2_retrieval_candidates.json`. Original dataset SHA-256: `553cd5a15e25f2ceccc6ed185221eba645080c93e5b91087560a91aa5961f365`; pinned dataset has 5,882 turns and 1,986 questions. The supplied caption field is `blip_caption` (1,226 string entries). Original data/submodule files were not changed.

Frozen before comparisons: seed 20261020; development conversations `conv-26`, `conv-30`, `conv-41`, `conv-43`, `conv-47`, `conv-49`; validation `conv-42`, `conv-44`, `conv-48`, `conv-50`. All 3,329 turns of the development histories are replayed. Fixed per-category sampling selects 116 questions (at most four per category per conversation), including 91 annotated answerable questions. All ten conversations were previously exposed; validation is held out only from this tuning and is not an untouched independent test set.

The small ladder predeclares sparse-only versus hash fusion; learned dense+sparse at existing weights; one sparse-dominant alternative; disabled versus lexical reranking; then the optional CPU cross-encoder. All use top-k 5, candidate pool 30, semantic representation and captions off. No top-k/pool/context/hop increase or rank-fusion implementation was introduced. Queries contain no answer/evidence labels; gold IDs enter only retrieval evaluation after a real controller response.

The existing large legacy investigation scripts were not rerun: the bounded `retrieval_development` entry point reuses canonical replay/runtime/retrieval and the Goal 1 compact audit, without unrelated report generation or synthetic fallback datasets. Identical replay stores are reused only by embedding/representation identity; the post-history identity-normalizer state is copied before each candidate's queries. Reports separate actual replay work, effective cold replay cost, query cost and cache provenance, and verify that QA does not change the memory population. Controlled-document ingestion is explicitly an ablation, not the normal-admission headline.

Real cached MiniLM preflight passed held-out paraphrase-vs-distractor comparisons: 0.798727 vs 0.106003 and 0.562012 vs 0.023116. Initial load plus checks took 21.964 seconds locally. This validates actual learned CPU execution and a small semantic sanity check; it does not establish LoCoMo answer quality. Raw preflight identity/results: `benchmarks/locomo/outputs/goal2/embedding_preflight.json` (generated artifact, not committed).

Measured mean annotation recall over the same 91 annotated answerable development questions (all 116 queries contribute to timing):

| Frozen candidate | Dense R@30 | Sparse R@30 | Fused R@30 | Returned R@5 | Mean query ms | Effective full replay s |
|---|---:|---:|---:|---:|---:|---:|
| L1 sparse only, lexical | unavailable | 0.412171 | 0.412171 | 0.217116 | 33.436 | 16.412 |
| L0 hash fusion, lexical | 0.077839 | 0.412171 | 0.317849 | 0.187812 | 44.379 | 16.412 |
| L2 learned fusion, lexical | 0.452464 | 0.412171 | 0.465368 | 0.291209 | 114.188 | 259.028 |
| L2 sparse-dominant, lexical | 0.452464 | 0.412171 | 0.483600 | 0.292208 | 111.527 | 259.028 |
| L2 learned fusion, disabled | 0.452464 | 0.412171 | 0.465368 | 0.260989 | 91.302 | 259.028 |
| L3 CPU cross-encoder | pending | pending | pending | pending | pending | pending |

The sparse-only trace records an observed empty dense list because that strategy is disabled; it is not a measured dense-retrieval zero. Reranking preserves all 30 pool candidates, so its pool recall equals fused recall; returned R@5 measures its ordering. Default weights are dense 0.4, sparse 0.3, temporal/graph 0.15 each; the single sparse-dominant alternative uses 0.2/0.5/0.15/0.15. Only dense/sparse execute under this matched `simple_rag` profile.

| Candidate | Embedding ms | Dense lookup ms | Sparse lookup ms | Fusion ms | Reranking ms |
|---|---:|---:|---:|---:|---:|
| L1 sparse only | 0 | 0 | 29.523 | 1.116 | 1.284 |
| L0 hash fusion | 0.412 | 4.700 | 33.839 | 1.645 | 1.776 |
| L2 learned fusion | 43.391 | 22.391 | 41.981 | 1.725 | 1.706 |
| L2 sparse-dominant | 43.391 | 21.728 | 39.695 | 1.774 | 1.833 |
| L2 disabled | 42.491 | 21.465 | 22.807 | 1.712 | 0.006 |

These are sequential local observations, not repeated controlled latency estimates. The 22.887 ms overall difference for disabled vs lexical includes sparse/cache/order variation; it must not be attributed entirely to reranking (measured reranking difference is about 1.7 ms). Model load is separate from replay; no system/cgroup peak-memory or Kaggle resource acceptance was measured. Actual replay work was 16.412 s for the first stable candidate and 259.028 s for the first learned candidate; other candidates reuse those histories. Their short wall times are not cold end-to-end costs.

Disposition: retain the safe production defaults and freeze `L2_learned_fusion` as a reproducible semantic candidate for later answer/resource gates. Sparse dominance improved pool recall slightly but barely changed returned recall; this one development sample does not justify a production weight change. Lexical ordering outperformed disabled ordering on returned recall here. No additional vector cache or rank-fusion method is justified by these measurements. No validation-conversation queries, Qwen inference, answer-score gain or normal-admission headline is claimed.

All 580 traces reconcile: exactly the frozen 116 IDs once per measured candidate, all six complete histories totaling 3,329 turns, unchanged post-query memory populations, top-k 5 and fused pool at most 30. L3 has no trace and is explicitly `externally_pending` because its exact snapshot is absent.

Generated artifacts live under `C:/PRIMA_next/tmp/person1-dev-p1/benchmarks/locomo/outputs/goal2/` and are intentionally ignored, not committed:

| Artifact/control | SHA-256 |
|---|---|
| `embedding_preflight.json` | `0a6b2f1555d6cf5d478c361304979340cdbbfd1c1b4e6bb97121e232726cb710` |
| `development/development_report.json` | `b8011e6b4e0d8c9f6760ac618094fdd71a07b87637ddfd89e2b05e3e682af81b` |
| `development/traces.jsonl` | `956b69fde2d95729218fa44d14289b4f4d10375e0148efe3229f3f025bb5d2a8` |
| `execution_identity.json` (base plus changed implementation/config hashes and artifact/control hashes) | `7f5bc84fb5e28dc6746daec386ed9bdada330a8ced00f5da2c712aa4678719a2` |
| committed `goal2_development_manifest.json` | `daf4445adab089683e236c1dc81f311684f78d0c2422ac6df90be6515c0cfcf3` |
| committed `goal2_retrieval_candidates.json` | `3bf91753aef668e3bd2d638d35452678a8fe925b8bb13aa3100f5549b8be1ea4` |

Hashes above use raw working-tree bytes, including the controls' original line endings. Loaded versions observed: PyTorch 2.14.1, sentence-transformers 6.1.0, transformers 5.18.0, huggingface-hub 1.33.0; no environment changes were made.

### Goal 2 validation ledger and remaining gates

Initial affected Goal 1/embedding/alignment suites: 82 passed. The first new suite found one synthetic admission test patching the runtime's unused importance-engine instance; it now patches the actual shared importance class. The corrected provenance/embedding/LoCoMo/shared integration selection passed 91 tests. The development split/ladder suite subsequently passed 49 tests; mypy initially identified two third-party/dictionary annotations, now corrected.

Final expanded affected regression suite: **290 passed in 81.46 s**, including Goal 1 audit/scorer cases and the shared integration replay/QA test (read-only test ownership). A final test-only review added four controlled-ingestion malformed-time cases and explicit non-unit-vector normalization coverage; the affected embedding/native replay recheck passed **38 tests in 27.09 s**. No measured inference implementation changed. Opt-in actual pinned CPU model smoke, including exact artifact/revision identity, finite values and unit norm: **1 passed, 1 deselected in 23.31 s**; deselected test is the separate Ollama smoke. The only real-cache warning is the upstream rename of `get_sentence_embedding_dimension`; native dimension still verifies correctly. Ruff: **all checks passed**. Strict affected mypy: **no issues in 50 source files**, with existing unused override-section notes. Staged and unstaged `git diff --check` passed, including all new files; the pinned external checkout is clean. An earlier long-running check's output was unavailable after context compaction; no live Python remained, so the final suite/smoke was rerun with saved logs rather than assuming success.

Reproduction commands (repository root; use the existing interpreter above):

```powershell
C:/PRIMA_next/.venv/Scripts/python.exe -m pytest -q tests/test_locomo_review_regressions.py tests/test_locomo_qa_correctness.py tests/test_embedding_backend.py tests/test_answer_prompt_identity.py tests/reasoning tests/retrieval tests/memory tests/evolution tests/lineage tests/workflow tests/reflection tests/runtime tests/evaluation tests/integration/test_runtime_pipeline.py
C:/PRIMA_next/.venv/Scripts/python.exe -m pytest -q tests/retrieval/test_locomo_native_replay.py tests/test_embedding_backend.py
$env:PRIMA_RUN_LEARNED_EMBEDDING_SMOKE='1'
C:/PRIMA_next/.venv/Scripts/python.exe -m pytest -q tests/test_production_embedding_smoke.py -k pinned_cpu
Remove-Item Env:PRIMA_RUN_LEARNED_EMBEDDING_SMOKE
C:/PRIMA_next/.venv/Scripts/python.exe -m ruff check benchmarks/locomo memory reasoning workflow uncertainty reflection runtime config/runtime_config.py tests/test_embedding_backend.py tests/test_production_embedding_smoke.py tests/test_locomo_qa_correctness.py tests/retrieval/test_locomo_native_replay.py
C:/PRIMA_next/.venv/Scripts/python.exe -m mypy config/runtime_config.py memory/embedding_backend.py memory/embedding_pipeline.py memory/embedding_rebuild.py memory/memory_repository.py memory/retrieval runtime workflow reasoning
C:/PRIMA_next/.venv/Scripts/python.exe -m benchmarks.locomo.retrieval_development --dataset benchmarks/locomo/external/data/locomo10.json --manifest tests/fixtures/locomo_review/goal2_development_manifest.json --candidates tests/fixtures/locomo_review/goal2_retrieval_candidates.json --output benchmarks/locomo/outputs/goal2/development
git diff --check
git diff --cached --check
git -C benchmarks/locomo/external status --porcelain
```

The measured output namespace already exists; use a fresh suffix when reproducing it. For Person 2, deserialize one complete candidate mapping with `RuntimeConfig.model_validate`, stage its immutable model revision, and supply `runtime_config=` to the canonical experiment. Shared preflight must invoke the selected auxiliary backend validation/load before admitting a campaign; missing files cannot trigger network loading or hash/lexical substitution. Exact candidate JSON, representation flag/hash, artifact identity and full typed configuration must travel with the run manifest, and persistent dimension changes require an explicitly rebuilt compatible collection. The cross-encoder's real-cache sanity/ranking and resource pilot remain pending after its artifacts are staged.

No Qwen answer generation, GPU allocation, full benchmark, policy promotion, source publication or dependency upgrade is part of Goal 2. Cross-encoder real loading/ranking, fresh integrated Kaggle timing/resources, normal-admission headline answer quality and frozen full evaluations remain later measured gates. Goal 3 starts only after this checkpoint is committed and its final checks pass. Rollback is a scoped Goal 2 revert or an explicitly selected prior safe configuration with a fresh run identity; never reuse incompatible checkpoints or reset `dev`.

## Goal 3 — bounded grounded attempts and question-preserving follow-ups

Actual base/predecessor: `6e6f968aa9c219b0ffe86d4d3fa2e4d534830f39`, the clean `dev-p1` HEAD on entry. Its parent is Goal 1 (`728709be17f7f8b08f67d01fb06e07c160bd612b`). Both checkpoints and their handoff/test ledgers were inspected before editing; current affected suites recheck their audit, scoring, provenance and retrieval contracts. Work remains in the isolated delivery clone. No Person 2 files, external sources, dependency/CI/notebook files or admission/reflection thresholds were edited.

### Requirement and caller coverage

| Requirement | Implementation/callers | Behavioral evidence |
|---|---|---|
| Preserve question subject/relation/time | `InformationNeedGenerator` reuses `QueryAnalyzer`, explicit missing-information/unresolved-entity fields, source spans and entity metadata; appends one supported bridge to the exact original question | Multi-hop controller/runtime fixtures; explicit unresolved entity and unsupported/generic-word cases; normalized duplicate checks |
| Remove both premature gates without claiming sufficiency | `ReasoningController._synthesize`, nullable `AnswerResult.answer_attempt_permitted`, `EvidenceAcquisitionController`, `AnswerGenerationController` | Low-overlap date evidence reaches one actual fake-provider call under default, single-pass, simple-RAG and full routes; status remains `need_more_evidence` |
| Guard useful-evidence attempts | Same two gates, post-compression permission/size validation and existing eligibility flag | Empty/missing/ineligible evidence, contradiction, hard error, exhausted call budget and expired deadline yield no speculative call; compressed final-prompt audit remains exact |
| Reserve time and share pre-call accounting | Existing reasoning limits plus `llm/request_budget.py`, execution context, runtime typed options and `LLMClient` retry loop | Answer reserve stops another retrieval while time remains; every retry/correction uses the same allowance; deadline reached during retrieval/provider retry suppresses later calls |
| Keep canonical generation single-owned | Acquisition calls `ReasoningController.answer` with no synthesis callback; workflow owns the single final model attempt | Four route tests observe exactly one model call; a bounded attempt cannot regenerate or retry the provider |
| Suppress late reflection/corrections | Reasoning advisor boundary, workflow reflection phase/controller and correction review | Expired deadline prevents advisor/evaluate calls and rejects correction advice; recovered phase failures remain distinguishable from unresolved hard errors |
| Preserve grounded outputs and audit | Existing factual schema, stricter citation validation, explicit source anchor in evidence text before compression, bounded factual policy | Model abstention accepted; unknown/unreported inline citation labels rejected; actual serialized and cited source IDs still satisfy Goal 1 fixtures |
| Export compatible consumer fields | Runtime response/compatibility adapter, LoCoMo question records/manifests and fixture | Missing old observations remain null; new attempt/budget fields survive prediction checkpoint roundtrip |

Caller searches covered `ReasoningController.answer`, synthesis callbacks, `InformationNeedGenerator`, verifier/stopping policies, all direct model/provider sends, runtime `execute`/`answer_question`, typed options, acquisition/compression, reflection, phase retries, correction routes, output validation and LoCoMo record/checkpoint consumers. Existing deterministic `reasoning.validate_controller` and `validate_reflection_integration` fixtures now use original-question-preserving queries. Shared integration/provider tests were run read-only.

### Query and answer behavior

A follow-up has the form `Which hospital treated Maya's friend last year? Source bridge: Cedar`, retaining the original question unchanged. A bridge is at most six words/80 characters and must occur in an anchored source span, either as an explicit unresolved/structured entity or after a supported linking expression such as `identifies`, `named` or `friend is`. Capitalization alone grants nothing. Acknowledgements/pronouns, punctuation/case/whitespace variants, unsupported entities and duplicate normalized queries are rejected. No useful expansion stops as `no_supported_expansion` and enters bounded answerability evaluation. Reasoning reflection and workflow query corrections use the same source-supported construction; they cannot replace the question with a generic standalone phrase. No planner/model call was added.

Lexical coverage retains its existing diagnostic/continuation threshold; the verifier and uncertainty/world-risk thresholds are unchanged. New controller results explicitly grant or deny `answer_attempt_permitted` independently of evidence sufficiency. At a normal retrieval stop, eligible nonempty evidence can receive one bounded attempt even with `NEED_MORE_EVIDENCE`. Contradiction, ambiguity, unanswerability, unresolved hard error, unavailable context/call/time budget still block it. Source time is included as labeled evidence data when absent from the source text, before word-budget compression; it is not a fabricated event date. The factual policy permits unambiguous anchored date resolution, complete supported lists and explicit multi-hop composition while requiring abstention when support is missing.

Acquisition defers model execution using an absent synthesis callback. Its final permission is checked after canonical compression. A standalone reasoning caller with an actual synthesis callback must fit the context budget before callback invocation. Workflow generation rechecks eligibility, status, final compressed size, shared budget and hard errors immediately before a call. A phase exception that successfully retries clears its unresolved-error marker while retaining historical error/audit records; an unresolved failure cannot use that recovery path.

One bounded low-coverage attempt disables provider retries even if the supplied generation config requested them, consumes `bounded_answer_attempts=1`, and cannot regenerate after invalid output or model abstention. Ordinary sufficient/conversation correction behavior remains bounded by the existing correction limits and the shared call allowance. All non-abstaining factual memory-grounded answers must report supplied citation labels; unknown labels and inline labels omitted from the evidence list fail parsing. The existing MODEL_ONLY profile remains the explicit benchmark baseline using user-supplied bounded transcript context; it does not receive new memory-evidence permission. Missing reasoning on other factual routes fails closed.

### Typed budget and diagnostic contract

| New/defaulted field | Default and meaning |
|---|---|
| `ExecutionOptions.max_llm_calls` | 4, nonnegative; zero explicitly disables model calls |
| `ExecutionOptions.time_budget_seconds` | 15.0, positive finite; one monotonic deadline for the workflow execution, not one deadline per reasoning pass |
| `ExecutionOptions.answer_reserve_seconds` | 3.0, finite/nonnegative; effective retrieval reserve is capped at 20% of the total deadline, so short requests remain bounded |
| `ReasoningBudget.answer_reserve_seconds` | 3.0; standalone factory additionally supports `PRIMA_REASONING_ANSWER_RESERVE_SECONDS` |
| `AnswerResult.answer_attempt_permitted` | null for old/unobserved results; explicit boolean for new decisions, without renaming/changing sufficiency status |
| `GenerationResult.blocked_reason` | null unless an actual call is prevented or a late result discarded, e.g. `time_budget`, `max_llm_calls`, `context_budget` |

Canonical callers use the typed `ExecutionOptions` fields; metadata cannot override these controls. The request-owned `RequestBudget` survives phase retries, acquisition retries, reflection corrections and model generations. Retrieval calls are charged before invocation, including failed attempts, against the shared retrieval cap. Existing hop/no-progress limits still apply within each reasoning pass. Context uses the existing approximate word-token accounting and compressor; the context limit, candidate pool and top-k are not increased.

An answer call reserves its allowance before dispatch. The client receives this first prepaid call through a request-local context variable propagated to the worker; each additional provider retry checks the same deadline and charges another allowance before `provider.send`. Actual provider timeout is clamped to the remaining request time. Rate-limit/provider/config failures can conservatively consume a reserved attempt even if no request reaches a server; they never undercharge retries. Standalone non-model synthesis callbacks retain the existing `llm_used=False` convention and release their reservation; production acquisition uses no callback at all. Direct `LLMClient` users outside the canonical request keep their existing standalone retry behavior.

The deadline prevents new retrieval/generation/reflection/correction work and discards late model answers. The async wait is bounded and provider HTTP timeouts are clamped, but a remote server's already-started computation cannot be forcibly cancelled by this client. A worker or synchronous adapter can still drain an in-flight request; integrated wall-time/resource gates remain required. This checkpoint does not claim server-side cancellation or a measured 15-second Qwen service time.

Runtime `output_data` and LoCoMo question records add:

- `answer_attempt_permitted`: true/false, or null when old/unobserved.
- `answer_attempt` version 1.0: `permitted`, `retrieval_stop_reason`, actual `attempted`, terminal `outcome`, and `blocked_reason`.
- `request_budget` version 1.0: declared call/time/reserve limits, shared `llm_calls`/`retrieval_calls`, current `remaining_seconds`, and `bounded_answer_attempts`.

Runtime also reports `reflection_suppressed_reason` when a workflow reflection phase is suppressed by the deadline. Existing `stop_reason` remains the retrieval stop reason; it is never replaced with the model answer/abstention outcome. `model_call_count` now includes every charged provider retry and generation/correction attempt, rather than only chat invocations. The final response's `request_budget` is authoritative; the reasoning diagnostic snapshot precedes workflow generation. Permission does not promise that a later call will still have time available.

Old records without these observations export null, not zero or a denial inferred from their outcome. Legacy `AnswerResult` instances with unobserved permission can retain the old `SUFFICIENT` route, but low-coverage permission is never inferred for them; an explicit new denial is authoritative. Evidence audit remains 1.0; question/metric schema stays 2.0 with additive fields and unchanged evidence unions/scorer meanings. LoCoMo manifests add `answer_contract_version=1.0`, so incompatible pre-A3 experiments cannot silently resume under the changed prompt/query/attempt policy. Goal 2 model/vector/representation identities and defaults are unchanged.

Committed consumer examples: `tests/fixtures/locomo_review/goal3_answer_contract.json`, SHA-256 (raw working-tree bytes) `4eb266402af6be397aa8a5ed1e2650c5317be77513f1ae025cce7aadc02ed618`. The new example is an actual canonical fake-provider output subset with observed timing, not a performance measurement. Tests verify old null observations, new low-coverage permission/status separation and checkpoint preservation. Person 2 should consume these fields directly and keep retrieval stop, permission, call accounting and model outcome distinct.

### Goal 3 validation ledger and remaining gates

Initial focused suite: 85 passed / six failed. Five fixtures encoded the old standalone/unsupported query behavior and were updated with supported question-preserving bridges. The sixth exposed an overbroad hard-error check; recovered phase errors now retain audit history without being treated as unresolved failures. Corrected existing focused suite: **91 passed**. New route/budget/citation/query tests then passed **121**, followed by the expanded LoCoMo selection at **149 passed**.

The first broad run passed 341 and failed the unchanged Goal 1 compression/prompt audit: an explicit denial had been computed before compression. The fix defers final context-size permission to the canonical compressor and still checks size before every actual provider call. Broad affected recheck: **342 passed in 41.10 s**. The focused missing-reasoning/legacy-null checks passed **160 in 14.24 s**; the explicit bounded-provider-retry selection passed **43 in 6.89 s**. The last review removed the redundant wall-clock stop check so all deadline decisions use the same monotonic request clock; the final affected route/LoCoMo/workflow/runtime recheck passed **161 in 14.02 s**. Strict affected mypy: **39 source files clean**, with only existing unused override-section notes. Ruff and staged/unstaged diff checks passed, including new files. Both deterministic reasoning validation entry points returned **PASS**. The pinned external checkout is clean. No failing check is being treated as acceptance evidence.

Commands from the delivery checkout, using the existing Python environment:

```powershell
C:/PRIMA_next/.venv/Scripts/python.exe -m pytest -q tests/test_locomo_review_regressions.py tests/test_locomo_qa_correctness.py tests/test_embedding_backend.py tests/test_answer_prompt_identity.py tests/reasoning tests/retrieval tests/memory tests/evolution tests/lineage tests/workflow tests/reflection tests/runtime tests/evaluation tests/integration/test_runtime_pipeline.py tests/integration/test_release_readiness.py
C:/PRIMA_next/.venv/Scripts/python.exe -m pytest -q tests/reasoning tests/test_locomo_review_regressions.py tests/test_locomo_qa_correctness.py tests/workflow tests/runtime tests/test_answer_prompt_identity.py
C:/PRIMA_next/.venv/Scripts/python.exe -m pytest -q tests/reasoning/test_grounded_answers.py
C:/PRIMA_next/.venv/Scripts/python.exe -m ruff check benchmarks/locomo memory reasoning workflow uncertainty reflection runtime llm tests/reasoning tests/workflow tests/runtime tests/test_locomo_review_regressions.py tests/test_locomo_qa_correctness.py tests/test_answer_prompt_identity.py
C:/PRIMA_next/.venv/Scripts/python.exe -m mypy reasoning workflow runtime llm
C:/PRIMA_next/.venv/Scripts/python.exe -m reasoning.validate_controller
C:/PRIMA_next/.venv/Scripts/python.exe -c "from reasoning.validate_reflection_integration import main; raise SystemExit(main())"
git diff --check
git diff --cached --check
git -C benchmarks/locomo/external status --porcelain
```

No matched L4 Qwen development comparison, answer-quality gain, Kaggle timing/resource claim, GPU allocation, model download or production promotion is made. These are deterministic implementation checks using fake providers, including the real `LLMClient` retry boundary with a synthetic provider. Goal 2's cached-model evidence remains its own retrieval-only measurement; it cannot establish quality or cost for the new A3 answer policy. Later matched L4 evaluation must reuse the frozen development controls/candidate identities, report answerable official/legacy F1 and unsupported adversarial outcomes together with actual call/time costs, and pass shared resource/identity gates before any promotion.

Goal 4 requires this committed checkpoint and passing A1–A3 checks, then owns evidence eligibility, admission/reflection policy investigation and final integrated review. Those policies were not started here. Person 2's final integration still waits for Person 1 Goal 4 in a separate pinned integration checkout. Rollback: scoped revert of this Goal 3 checkpoint, preserving prior checkpoints/artifacts and using a fresh run identity; no reset, storage migration or archive rewriting.
