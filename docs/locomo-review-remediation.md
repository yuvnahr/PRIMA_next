# Person 1 LoCoMo/runtime remediation handoff

## Checkpoint and provenance

Goal 1 (A1): implementation complete; mandatory deterministic, caller, lint and diff checks passed. Goals 2–4 have not started.
Actual base: `c25a178d6308bc48b3a5e44d9519316a5179c499` (the planning snapshot); no intervening tracked implementation changes.
Predecessor: none. The completed checkpoint SHA is reported in the completion message, not inside its own commit.

Delivery checkout: `C:/PRIMA_next/tmp/person1-dev-p1`, branch `dev-p1`.
This is a clean local clone of the assigned branch because the original `C:/PRIMA_next` checkout contained unrelated deletions/untracked files and a dirty LoCoMo-v2 submodule. Those files, its index and branch pointer were left alone. No push, PR, merge or production execution is authorized by this checkpoint.
The interpreter is `C:/PRIMA_next/.venv/Scripts/python.exe`, Python 3.14.8. Sandbox execution could not access its base installation; approved execution outside that boundary uses the same existing environment, without dependency changes.

Scope source: `Person_1_LoCoMo_Runtime_Implementation.md`, SHA-256 `43bf76b274adba04947e630eef9fea767d493c2b4976a72090656b4b7f914dab`.
The attachment invokes Goal 1 only. The newer four-goal operating contract supersedes the source plan's intermediate PR/merge workflow.
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
