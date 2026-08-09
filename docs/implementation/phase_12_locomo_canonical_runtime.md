# Phase 12 — LoCoMo Canonical Runtime Migration

## Outcome

LoCoMo now measures the canonical runtime instead of combining the generic benchmark runner with a separate QA adapter. Every conversation turn and every question is represented by a typed `PrimaRequest` and sent through `PrimaRuntime.execute()`.

The migration adopts the Phase 10 common artifact store and the Phase 11 paired-profile pattern. Production packages do not import LoCoMo, benchmark gold is never added to prompts or runtime metadata, and no historical result artifact or source dataset was changed.

## Profiles and ingestion policies

Paired runs execute one deterministic question selection through:

- `model_only` with bounded, timestamped conversation context and controlled document ingestion;
- `simple_rag` with the simple vector-memory baseline;
- `prima_full` with normal PRIMA conversation admission.

The policies are explicit manifest fields:

- `controlled_document_ingestion` sends turns through `document_ingestion/ingestion_only`; model-only questions receive only the configured bounded transcript;
- `simple_vector_memory_baseline` sends turns through `document_ingestion/ingestion_only`, then answers with the canonical dense+sparse `simple_rag` route;
- `normal_prima_admission` sends turns through `conversation/prima_full`, preserving workflow-owned admission, state, memory, and maintenance behavior.

Each conversation receives an isolated benchmark runtime and repository. Paired and threaded runs reuse one injected heavyweight model client. Timestamps are normalized and included in both bounded context and ingested turn text so temporal source information reaches the model without exposing gold answers or evidence annotations.

## Validation

The adapter now rejects malformed roots, records, conversations, numbered sessions, turns, questions, categories, evidence annotations, evidence references, and timestamps with conversation/question locations in the error message.

Supported timestamps include ISO date/time values and the known LoCoMo day/month and month/day 12-hour forms. Source timestamps without an offset are normalized as UTC. Scalar numeric answers in the source dataset are preserved as strings; booleans and structured answer values remain invalid.

Strict validation intentionally does not repair benchmark gold. The bundled `locomo10.json` contains malformed annotations, including `conv-42` question 58 referencing an absent `D10:19`. Loading that source fails with the exact conversation, question, and evidence ID. A corrected dataset revision is required for a full run; this phase does not delete or rewrite invalid gold evidence.

## Metrics

Core evaluation runs without ROUGE-L or BERTScore. The documented normalization policy is the established SQuAD policy: lowercase, remove punctuation and the articles `a`, `an`, and `the`, then collapse whitespace. Meaning-bearing conjunctions remain. Exact match compares the complete normalized strings, preserving order and multiplicity. Token F1 uses `Counter` intersection, so repeated tokens retain multiplicity.

ROUGE-L uses the built-in deterministic implementation only when requested. BERTScore remains optional, reports capability status, and receives the configured device and batch size. The existing benchmark preflight reports built-in ROUGE-L and optional `bert_score` availability without importing or installing models.

The permissive memory-hit proxy is removed. LoCoMo now reports:

- candidate evidence-ID recall from the fused retrieval pool, with dense+sparse fallback when no fused stage exists;
- final evidence-ID recall from the evidence delivered to answer generation;
- answer-token coverage as a separate diagnostic;
- memory-admission recall against annotated source-turn IDs;
- unique memory growth and maintenance completion;
- category-wise EM/F1, category-2 temporal versus non-temporal results, and single-session, multi-session, or unannotated results;
- typed abstention/category-5 accuracy;
- execution-failed questions separately from evaluation failures.

Every declared failure subcode has a tested branch. Empty successful responses are execution failures; ordinary wrong answers, retrieval misses, and evidence-selection losses remain evaluation failures. Category-5 typed abstention is treated as a correct abstention rather than an answer-parse failure. Category totals and the execution/evaluation failure partition are checked before finalization.

## Artifacts, resume, and dataset scope

Each question is checkpointed immediately through `BenchmarkArtifactStore`. Interruption leaves a partial manifest, completed question IDs are suppressed on resume, and changed dataset/model/generation/profile/policy/selection configuration is rejected by the common manifest fingerprint.

The one-conversation default is labeled `preview`. Unlimited execution is rejected unless `--full-dataset` is explicit; that flag cannot be combined with conversation or question limits. Paired manifests must contain the same selected question IDs.

Question records, conversation diagnostics, metrics, predictions, failures, checkpoints, summary, and manifest artifacts are schema-versioned. Memory growth, admission, retrieval stages, reflection calls, model usage, latency, and maintenance state remain visible in raw records and aggregate metrics.

## Focused verification

```text
venv\Scripts\python.exe --version
Python 3.10.11

venv\Scripts\python.exe -m pytest -q tests/test_locomo_qa_correctness.py tests/benchmarks/test_common_artifacts.py tests/benchmarks/test_preflight.py tests/retrieval/test_phase07_retrieval_alignment.py
36 passed in 30.28s

venv\Scripts\python.exe -m compileall -q benchmarks/locomo runtime/prima_runtime.py tests/test_locomo_qa_correctness.py tests/benchmarks/test_common_artifacts.py tests/retrieval/test_phase07_retrieval_alignment.py
Exit 0; no output.

venv\Scripts\python.exe -m ruff check benchmarks/locomo benchmarks/common/interfaces.py runtime/prima_runtime.py tests/test_locomo_qa_correctness.py tests/benchmarks/test_common_artifacts.py tests/retrieval/test_phase07_retrieval_alignment.py
All checks passed.

venv\Scripts\python.exe -m mypy benchmarks/locomo/adapter.py benchmarks/locomo/evaluate.py benchmarks/locomo/experiment.py runtime/prima_runtime.py
Success: no issues found in 4 source files.

Bundled timestamp validation: all 288 timestamp values (287 unique strings) parsed through `parse_timestamp`.
Bundled strict dataset validation: stopped at `conv-42` question 58 because gold evidence `D10:19` is absent from that conversation.
```

## Final quality gates

The repository-wide CI matrix was run once after implementation and documentation completed:

```text
venv\Scripts\python.exe -m benchmarks.preflight --json
Exit 0; schema 1.0; rouge_l, bertscore, and goemotions_encoder available.

venv\Scripts\python.exe -m compileall -q -x "(^|[\\/])(\.git|\.venv|venv|external)([\\/]|$)" .
Exit 0; no output.

venv\Scripts\python.exe -m pytest -q
292 passed, 1 skipped in 128.69s.

venv\Scripts\python.exe -m ruff check .
All checks passed.

venv\Scripts\python.exe -m mypy .
Success: no issues found in 221 source files. The run emitted a non-failing unused optional-module override note.

venv\Scripts\python.exe -m bandit -q -c bandit.yaml -r .
Exit 0; only nosec-without-failed-test warnings for the bounded Git probes in the HotpotQA and LoCoMo experiments.

venv\Scripts\python.exe scripts\run_xenon.py
Exit 0; production source roots only.
```

The unnecessary optional-module mypy override was then removed in favor of dynamic import after capability preflight. Per the one-CI-pass rule, only focused verification was repeated:

```text
venv\Scripts\python.exe -m pytest -q tests/test_locomo_qa_correctness.py tests/benchmarks/test_preflight.py
17 passed in 27.34s.

venv\Scripts\python.exe -m ruff check benchmarks/locomo/evaluate.py tests/test_locomo_qa_correctness.py
All checks passed.

venv\Scripts\python.exe -m mypy benchmarks/locomo/adapter.py benchmarks/locomo/evaluate.py benchmarks/locomo/experiment.py runtime/prima_runtime.py
Success: no issues found in 4 source files.
```

## Changed files

- `README.md`
- `benchmarks/locomo/{__init__.py,adapter.py,evaluate.py,experiment.py}`
- removed `benchmarks/locomo/runner.py`
- `runtime/prima_runtime.py`
- `tests/test_locomo_qa_correctness.py`
- `tests/benchmarks/test_common_artifacts.py`
- `tests/retrieval/test_phase07_retrieval_alignment.py`
- `docs/implementation/{STATUS.md,phase_12_locomo_canonical_runtime.md}`

## Risks and deferrals

- The bundled LoCoMo source cannot pass strict full-dataset validation until its malformed evidence annotations are corrected in an authorized dataset revision. No gold repair is inferred or applied here.
- Model-quality scores were not regenerated and no full dataset was executed.
- BERTScore behavior is covered with capability and mocked device/batch tests; no model assets were downloaded.
- GoEmotions remains the final benchmark migration phase before cross-benchmark complete-wrapper attribution is valid.
