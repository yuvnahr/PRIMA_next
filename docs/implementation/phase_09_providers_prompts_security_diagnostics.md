# Phase 09 — Providers, Prompts, Security and Diagnostics

## Outcome

Phase 09 preserves the workflow-owned canonical runtime and removes the unused answer-synthesis path that could create a separate client and fallback policy. The runtime now owns one injected provider session and rate limiter, while each request carries an immutable typed generation contract.

`Draft.png` remains the target architecture, not the current call graph. This phase strengthens the existing model-execution and diagnostic blocks; it does not migrate benchmark boundaries or claim new benchmark scores.

## Contract and behavior changes

- Added immutable `GenerationConfig`, `StructuredOutputMode`, `FallbackPolicy`, and `ProviderCapabilities` contracts.
- Configuration covers provider, model, temperature, top-p, optional top-k/repeat penalty/seed, output-token limit, timeout, retries, structured-output mode, and fallback policy.
- Runtime construction captures settings once. Provider sends and workflow execution no longer read mutable environment variables for generation behavior.
- Provider adapters preserve supported system prompts and schemas and reject unsupported fields explicitly. Retries are owned by the reusable client and governed only by the request config.
- OpenAI, OpenAI-compatible/LM Studio, Anthropic Messages, and native Ollama payload behavior is explicit. Tests intercept HTTP payloads; no live provider calls occur.
- `PromptBuilder` separates trusted system policy from untrusted user input, retrieved memory, tool results, trusted runtime context, and output schema.
- Provider/parse failures follow a configured typed policy: `fail`, `abstain`, or evidence-backed `extractive`. Public abstention is a typed outcome, never a JSON string.
- The legacy `_synthesize_evidence` path and its duplicate prompt, fallback, environment, logging, and diagnostics helpers were removed.
- One diagnostics assembler serves all task routes. It reports measured latency, retrieval/reflection/model-call counts, token usage, provider capabilities, fallback classification, and complete trace-event counts.
- Standard mode retains compact counts and decisions; diagnostic mode retains every workflow event for the request.
- Recursive redaction covers secret-like fields, authorization headers, endpoint query credentials, and raw prompts when configured.

## Architecture ownership

- `PrimaRuntime` owns the reusable `LLMClient`, default `GenerationConfig`, and public diagnostics assembly.
- `PrimaRequest` may override generation configuration without mutating process-global state.
- `AnswerGenerationController` remains the only production answer-generation owner and consumes the injected client.
- Provider adapters own payload translation and capability validation, not orchestration or fallback policy.
- Benchmarks remain outside production packages and are not migrated in this phase.

## Verification

Focused tests cover provider payload parity, capability failures, reusable-client identity, retries, prompt and retrieved-memory injection isolation, typed fallback classification, diagnostics counters/full traces, redaction, canonical runtime compatibility, LoCoMo response parsing, and GoEmotions provider payloads.

Focused results during implementation:

```text
venv\Scripts\python.exe -m pytest -q tests/test_llm_structured_output.py tests/runtime/test_phase09_diagnostics.py tests/integration/test_runtime_pipeline.py
16 passed in 20.89s

venv\Scripts\python.exe -m mypy llm runtime workflow/answer_generation.py security/redaction.py
Success: no issues found in 19 source files

venv\Scripts\python.exe -m pytest -q tests/workflow/test_phase06_correction_loop.py::test_pre_execution_reflection_changes_query_evidence_and_answer
1 passed in 20.74s

venv\Scripts\python.exe -m pytest -q tests/test_llm_structured_output.py
6 passed in 0.56s
```

The first repository-wide pytest pass exposed the direct-workflow typed-config compatibility gap and stopped CI as designed:

```text
267 passed, 1 failed, 1 skipped in 308.73s
Failure: test_pre_execution_reflection_changes_query_evidence_and_answer
Cause: direct workflow construction lacked a captured default GenerationConfig.
Resolution: PrimaWorkflow captures one typed default from its injected client; the focused regression test passed.
```

Final quality-gate results after the fix:

```text
venv\Scripts\python.exe -m benchmarks.preflight --json
schema_version=1.0; rouge_l=true; bertscore=true; goemotions_encoder=true; EXIT_CODE=0

venv\Scripts\python.exe -m compileall -q -x '(^|[\\/])(\.git|\.venv|venv|external)([\\/]|$)' .
EXIT_CODE=0

venv\Scripts\python.exe -m pytest -q
268 passed, 1 skipped in 259.68s; EXIT_CODE=0

venv\Scripts\python.exe -m ruff check .
All checks passed!; EXIT_CODE=0

venv\Scripts\python.exe scripts/run_xenon.py
Production source roots only; EXIT_CODE=0

venv\Scripts\python.exe -m mypy .
Success: no issues found in 218 source files; EXIT_CODE=0

venv\Scripts\python.exe -m bandit -q -c bandit.yaml -r .
One existing nosec warning at benchmarks/hotpotqa/experiment.py:67; EXIT_CODE=0
```

The initial sandboxed CI launch returned exit 101 before any gate ran because the base Python executable is outside the workspace. The same requested pass was resumed with the approved virtual-environment interpreter.

## Intentionally deferred risks

- Benchmark runners still need migration to the canonical typed boundary before scores can be attributed to one complete PRIMA wrapper.
- Provider adapters use HTTP chat/Messages endpoints rather than vendor SDKs; future endpoints must declare their own capability contract.
- Raw prompts can be retained only when callers explicitly select diagnostic mode and disable prompt redaction; production policy should keep redaction enabled.

## Changed files

- `config/settings.py`
- `docs/architecture/ADR_CANONICAL_RUNTIME.md`
- `docs/implementation/{STATUS.md,phase_09_providers_prompts_security_diagnostics.md}`
- `llm/{generation_config.py,llm_client.py,llm_types.py,prompt_builder.py,provider.py,response_parser.py}`
- `pyproject.toml`
- `reasoning/validate_controller.py`
- `runtime/{__init__.py,contracts.py,prima_runtime.py}`
- `security/redaction.py`
- `workflow/{answer_generation.py,prima_workflow.py}`
- `tests/runtime/test_phase09_diagnostics.py`
- `tests/{test_goemotions_upgrade.py,test_llm_structured_output.py,test_locomo_qa_correctness.py}`

The pre-existing `benchmarks/locomo-v2/external` submodule marker remains untouched. Test-regenerated historical evaluation JSON was restored to `HEAD` and is not part of this phase.
