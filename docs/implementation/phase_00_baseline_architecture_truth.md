# Phase 00 — Baseline and Architecture Truth

## Outcome

Phase 00 changed documentation only. It froze the current execution reality, created closure IDs for later phases and made no production or benchmark behavior change.

The central finding is verified: `C:\PRIMA_integrated\Draft.png` is a target architecture, not the current call graph. Today, workflow turns, factual QA, direct ingestion and emotion classification use different execution boundaries.

## Baseline

- Branch: `dev`
- Commit: `8e45914e1b4b0ce66236be3b4c041001fb1fc5e8`
- Pre-existing changes: `codex_rules.txt` modified by the user-requested rules update; dirty `benchmarks/locomo-v2/external` submodule.
- Prior phase: none; Phase 00 is the first implementation ledger phase.
- Python used for valid gates: `C:\PRIMA_next\venv\Scripts\python.exe`, Python 3.10.11.

## Exact current call graphs

### A. `PrimaRuntime.process()`

```text
process(user_input)
└─ asyncio.run(process_async(user_input))
   ├─ ExecutionContext(user_input, cognitive_state)
   ├─ PrimaWorkflow.run
   │  └─ TaskRouter.route → AFFECT → MEMORY_RETRIEVAL → PLANNING
   │     → REFLECTION → ACTION → OUTPUT
   ├─ _persist_turn_memory → MemoryImportanceEngine → MemoryNote.create
   │  → MemoryRepository.add
   ├─ _build_result → output.text or user_input
   ├─ RuntimeContext update
   └─ _log_runtime_result
```

### B. `PrimaRuntime.process_async()`

`process_async` is the body shown under `process` above. It owns workflow invocation, memory admission, result shaping and runtime logging. Its output phase currently echoes the input.

### C. `PrimaRuntime.answer_question()`

```text
answer_question(question)
├─ ReasoningRequest
├─ ReasoningController.answer
│  ├─ RetrievalController.retrieve
│  ├─ RuntimeContextBuilder.build
│  ├─ _synthesize_evidence
│  │  ├─ LLMClient.chat, when evidence/model are available
│  │  └─ extractive fallback
│  ├─ ConfidenceEvaluator
│  └─ ReasoningReflectionAdapter (ABSTAIN or NO_ACTION only)
├─ partial answer_diagnostics update
└─ answer log with hard-coded latency/reflection fields
```

There is no call to `PrimaWorkflow.run`.

### D. `PrimaRuntime.ingest_document()`

```text
ingest_document(document)
└─ MemoryNote.create(type=SEMANTIC)
   └─ MemoryRepository.add
```

There is no workflow, typed route, admission policy, commit event or diagnostic lifecycle.

### E. `PrimaRuntimeAdapter.process_turn()`

```text
process_turn(text)
├─ document_ingestion=True  → PrimaRuntime.ingest_document → synthetic "Document ingested" response
└─ document_ingestion=False → PrimaRuntime.process → _response_from_runtime
```

### F. `PrimaRuntimeAdapter.answer_question()`

```text
answer_question(question)
└─ PrimaRuntime.answer_question
   └─ AnswerResult → AgentResponse
```

### G. HotpotQA

```text
run_hotpotqa_experiment
└─ run_sample
   ├─ PrimaRuntimeAdapter(document_ingestion=True)
   └─ GenericBenchmarkRunner.run
      ├─ each supplied context passage → process_turn → direct ingest_document
      ├─ question → adapter.answer_question → separate QA reasoning/LLM path
      ├─ gold/supporting facts attached to BenchmarkResult after inference
      └─ metrics/artifact/checkpoint writes
```

HotpotQA uses the dataset’s supplied context. It is not open-domain Wikipedia retrieval.

### H. LoCoMo

```text
run_locomo_experiment
└─ run_single_conversation
   ├─ PrimaRuntimeAdapter(document_ingestion=False)
   └─ GenericBenchmarkRunner.run
      ├─ each conversation turn → process_turn → workflow echo + episodic admission
      ├─ each question → adapter.answer_question → separate QA reasoning/LLM path
      └─ evaluator/artifact writes
```

The workflow output is not the QA answer path.

### I. GoEmotions

```text
run_goemotions_experiment
├─ select system
│  ├─ qwen_zero_shot → LLMClient
│  ├─ qwen_zero_shot_telemetry → LLMClient + independent affect telemetry
│  ├─ prima_qwen → Qwen + legacy affect
│  ├─ encoder → GoEmotionsEncoder
│  ├─ hybrid → encoder + optional Qwen
│  └─ prima_goemotions → encoder → decision controller → DynamicAffectEngine
└─ labels → metrics/artifacts
```

No GoEmotions system enters `PrimaRuntime` or `PrimaWorkflow`.

## Verified findings

- Workflow output echo: output returns `context.user_input`; result shaping repeats it.
- QA workflow bypass: `answer_question` owns a separate retrieval/reasoning/LLM loop.
- Separate benchmark boundaries: HotpotQA stitches direct ingestion to QA; LoCoMo stitches workflow turns to QA; GoEmotions uses classifier-specific systems.
- Ineffective reflection advice: the QA adapter cannot emit any revised-query action accepted by the controller.
- Disconnected graph/world/uncertainty/maintenance: graph is absent from runtime defaults; world and uncertainty values are read but never produced; maintenance engines lack production scheduling/subscribers.
- Procedural-memory absence: no procedural `MemoryType` exists.
- In-memory runtime default: `PrimaRuntime` constructs `InMemoryMemoryRepository` by default.
- Lexical reranker fallback: cross-encoder is disabled by default and failures fall back to lexical scoring.
- Incomplete diagnostics: no authoritative activated/skipped subsystem list; richer QA helper is unused; logged latency/reflection values are inaccurate constants.
- Benchmark checkpoint inconsistencies: Hotpot ignores `checkpoint_every`, only Hotpot resumes, and JSON checkpoint/manifest artifacts lack schema versions.

The full evidence and closure tests are in `ARCHITECTURE_TRUTH_MATRIX.md` and `STATUS.md`.

## Hypotheses deliberately not promoted to facts

- No disconnected component is assumed to improve accuracy merely because its class exists.
- No current benchmark result is attributed to the complete PRIMA wrapper.
- The trained TF-IDF/logistic classifier is not evidence that PRIMA improves Qwen.
- The target does not require irrelevant QA phases for emotion classification or ingestion.

## Repository statistics

| Metric | Value |
|---|---:|
| Tracked files | 564 |
| Python files | 341 |
| Production Python files | 228 |
| Benchmark Python files | 45 |
| Test Python files | 68 |
| Python lines | 25,937 |
| Production Python lines | 17,778 |
| Benchmark Python lines | 4,987 |
| Test Python lines | 3,172 |
| Loose Git objects | 3,251 (26.58 MiB) |
| Packed Git objects | 317 in 2 packs (18.44 MiB) |
| Git garbage | 0 |

Counts use tracked `*.py` files; “production” excludes `tests/` and `benchmarks/`.

## Quality gates

| Command | Result |
|---|---|
| `venv\Scripts\python.exe -m compileall -q .` | **Failed**, exit 1 in 84.1 s. Two syntax errors are inside the pre-existing dirty `benchmarks/locomo-v2/external` submodule; one is a Python-3.12-only Torch fixture under `venv/Lib/site-packages`. No owned source error was reported. |
| `venv\Scripts\python.exe -m pytest -q` | **Passed**, exit 0: `212 passed in 152.80s (0:02:32)`. |
| `venv\Scripts\python.exe -m ruff check .` | **Passed**, exit 0: `All checks passed!`. |
| `venv\Scripts\python.exe -m mypy .` | **Passed**, exit 0: `Success: no issues found in 341 source files`. |
| `venv\Scripts\python.exe -m bandit -c bandit.yaml -r .` | **Passed**, exit 0: no issues; 22,619 lines scanned; 14 disabled checks; 0 files skipped. |

Exact `compileall` errors:

```text
benchmarks/locomo-v2/external/benchmark_toolkit/opencode/runners/opencode_build_locomo_lance.py:53
SyntaxError: f-string: unmatched '['

benchmarks/locomo-v2/external/scripts/generation/generate_replacement_questions.py:24
SyntaxError: f-string: unmatched '('

venv/Lib/site-packages/torch/testing/_internal/py312_intrinsics.py:11
SyntaxError: invalid syntax
```

Bandit additionally warned that a `# nosec B603` in `benchmarks/hotpotqa/experiment.py:66` did not correspond to a failed enabled test; it reported no issue.

## Files changed

- `docs/implementation/ARCHITECTURE_TRUTH_MATRIX.md`
- `docs/implementation/STATUS.md`
- `docs/implementation/phase_00_baseline_architecture_truth.md`

`codex_rules.txt` was already modified for the user-requested architecture guardrails before this phase. The external submodule dirtiness also predates Phase 00.

## Behavior and contract changes

None. This phase adds documentation only.

## Risks and deferred work

- `GATE-001` blocks a clean literal `compileall -q .` baseline until the gate excludes environment/vendor trees or the external submodule is separately repaired. Phase 00 does not alter either target.
- Current benchmark results remain path-specific and must not be presented as one complete-wrapper measurement.
- Production refactoring is intentionally deferred to the owner phases in `STATUS.md`.
- Stop here. Do not begin Phase 01 until the user reviews the ledger and the compile-gate policy.

