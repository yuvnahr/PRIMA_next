# Implementation Status

Phase 00 baseline: 2026-08-02. Updated through Phase 02. `Open` means verified debt, not incomplete work in the latest phase. Owner labels must preserve the ID or record why ownership changed.

| ID | Severity | Kind | Finding and evidence | Owner phase | Acceptance test | Status |
|---|---|---|---|---|---|---|
| ARCH-001 | Critical | Verified fact | Public execution was split across `process`, `answer_question` and `ingest_document`; `execute(PrimaRequest)` now exists but legacy callers are not migrated. | Phase 02 contract; later workflow migration | All task kinds execute through one workflow lifecycle; legacy methods are thin adapters. | In progress |
| ARCH-002 | Critical | Verified fact | Workflow output echoes input (`workflow/prima_workflow.py:183`; `runtime/prima_runtime.py:386-388`). | Phase 03 — execution/output | A model-only workflow test proves output comes from the injected executor and differs from a sentinel input. | Open |
| ARCH-003 | Critical | Verified fact | `answer_question` calls `ReasoningController` directly and never calls the workflow. | Phase 03 — execution/output | QA adapter test proves exactly one workflow lifecycle and one executor call. | Open |
| ARCH-004 | High | Verified fact | `ingest_document` directly writes semantic memory and bypasses workflow/events. | Phase 02 — workflow routing | Ingestion test proves one public-boundary call, explicit skipped phases and a committed memory result. | Open |
| ROUTE-001 | High | Verified fact | Router had phase lists/untyped metadata, not typed task kinds or execution profiles. | Phase 02 — canonical contract | Tests cover all 25 task/profile decisions and explicit planned/executed/skipped components. | Closed |
| REFL-001 | High | Verified fact | QA reflection adapter emits only `ABSTAIN` or `NO_ACTION`; controller accepts only revised-query actions (`reflection/reasoning_reflection_adapter.py:17-21`; `reasoning/controller.py:83-109`). | Phase 04 — bounded reasoning | A low-confidence test demonstrates one advised query and a bounded second retrieval. | Open |
| REFL-002 | High | Verified fact | Workflow reflection records a decision but cannot retrieve/replan/correct. | Phase 04 — bounded reasoning | State-machine tests prove bounded retry, terminal success and retry exhaustion. | Open |
| SUBSYS-001 | Medium | Verified fact | Graph traversal is absent from runtime retrieval defaults; graph reasoning is not production-reachable. | Phase 05 — memory/graph | A profile test proves graph calls when enabled and an explicit skip when disabled. | Open |
| SUBSYS-002 | High | Verified fact | World model exists but no workflow stage invokes it; action only reads unset metadata. | Phase 04 — bounded reasoning | Full-profile test proves world simulation output reaches uncertainty/action diagnostics. | Open |
| SUBSYS-003 | High | Verified fact | Uncertainty estimator exists but no workflow stage invokes it. | Phase 04 — bounded reasoning | Threshold tests prove execute versus reflect transitions and bounded termination. | Open |
| SUBSYS-004 | Medium | Verified fact | Consolidation/abstraction/forgetting exist without production scheduling or event subscribers. | Phase 05 — memory/graph | Commit emits versioned events; maintenance consumer tests prove each configured handler call. | Open |
| MEM-001 | Medium | Verified fact | Procedural memory is absent from `MemoryType`. | Phase 05 — memory/graph | Contract and repository tests prove procedural storage/retrieval, or the target is explicitly retired. | Open |
| MEM-002 | High | Verified fact | Runtime defaults to `InMemoryMemoryRepository` (`runtime/prima_runtime.py:50`), despite persistence claims. | Phase 05 — memory/graph | Default/deployed configuration test proves the documented store; tests inject in-memory explicitly. | Open |
| RETR-001 | Medium | Verified fact | Reranker defaults to lexical scoring; cross-encoder is optional and load failure silently falls back (`memory/retrieval/reranker.py:35-100`). | Phase 05 — memory/graph | Diagnostics contract test identifies requested/actual reranker and fallback reason. | Open |
| DIAG-001 | High | Verified fact | Canonical results omit activated/skipped subsystems; richer QA diagnostics helper is unused; QA log hard-codes zero latency/false reflection. | Phase 06 — diagnostics | Every route returns schema-versioned phase/call/latency/fallback diagnostics verified against spies. | Open |
| BENCH-001 | Critical | Verified fact | HotpotQA, LoCoMo and GoEmotions exercise different boundaries; GoEmotions never enters `PrimaRuntime`. | Phase 07 — benchmarks | All runners use only the public runtime contract; production packages import no benchmark modules. | Open |
| BENCH-002 | Medium | Verified fact | Hotpot `checkpoint_every` is ignored, only Hotpot resumes, and checkpoint/manifest JSON lacks schema versions. | Phase 07 — benchmarks | Resume/cadence tests cover all supported runners and reject incompatible schema versions. | Open |
| BENCH-003 | High | Verified fact | Current results cannot be attributed to one complete wrapper; Hotpot uses supplied context and GoEmotions classifier variants do not measure Qwen-wrapper uplift. | Phase 07 — benchmarks | Reports identify exact route/profile/components/data regime and prohibit unsupported attribution text. | Open |
| CFG-001 | Medium | Verified fact | `pyproject.toml` targeted Python 3.10 and mypy used blanket `ignore_errors`. | Phase 01 — quality gates | CI/Ruff/mypy target 3.12; strict core modules pass without `ignore_errors`. | Closed |
| DOC-001 | Medium | Verified fact | README’s single-workflow and persistence claims exceed current reachability/defaults. | Phase 08 — tooling/docs | Documentation assertions match executable architecture tests and deployed defaults. | Open |
| GATE-001 | Medium | Verified fact | Literal `compileall -q .` traversed external/venv sources and failed outside owned code. | Phase 01 — quality gates | Python 3.12 owned-source compile gate explicitly excludes Git, venv and external trees and passes. | Closed |
| GATE-002 | Medium | Verified fact | Raw `xenon .` descended into non-production trees and did not finish within three minutes; the existing runner also included benchmark/evaluation roots. | Phase 02 — quality gates | CI/rules use `scripts/run_xenon.py`; a test proves its allowlist contains only production PRIMA roots. | Closed |
| DEP-001 | High | Verified fact | Runtime, development, benchmark, semantic-metric and encoder dependencies were mixed in one default requirements file. | Phase 01 — quality gates | Five explicit dependency groups exist; default install excludes semantic metrics and encoder/training extras. | Closed |
| METRIC-001 | High | Verified fact | LoCoMo imported ROUGE and BERTScore packages during normal evaluation, making optional metrics implicit test requirements. | Phase 01 — quality gates | Internal ROUGE-L is deterministic; BERTScore is opt-in and missing packages return explicit status. | Closed |
| HYGIENE-001 | Medium | Verified fact | Generated caches could pollute recursive gates; source-tracking policy was not executable. | Phase 01 — quality gates | Test proves cache paths are ignored and no cache artifact is tracked by the main repository. | Closed |
| LINT-001 | Low | Verified fact | Ruff globally exempted silent exception handling, exception chaining and compact-statement rules. | Phase 01 — quality gates | Five broad exemptions are removed or narrowed; remaining Python-upgrade deferrals are documented by rule. | Closed |
| CONTRACT-001 | Critical | Verified fact | No versioned typed canonical request/response API or serialization contract existed. | Phase 02 — canonical contract | Request/response JSON round trips; unknown/invalid input is rejected; async API returns typed response. | Closed |
| ROUTE-002 | High | Verified fact | No authoritative validity decision existed for every task/profile combination. | Phase 02 — canonical contract | Route matrix has 25 deterministic cells, nine valid plans and sixteen explicit rejections. | Closed |
| SYNC-001 | High | Verified fact | Sync entry points could call `asyncio.run` without an explicit active-loop contract. | Phase 02 — canonical contract | `execute_sync` works without a loop and directs active-loop callers to await `execute`. | Closed |

## Phase ownership map

| Later phase | Finding IDs |
|---|---|
| Later workflow-migration phase | ARCH-001 closure, ARCH-004 |
| Phase 01 — quality gates | CFG-001, GATE-001, DEP-001, METRIC-001, HYGIENE-001, LINT-001 |
| Phase 02 — canonical contract | ARCH-001 (partial), ROUTE-001, CONTRACT-001, ROUTE-002, SYNC-001, GATE-002 |
| Phase 03 — execution/output | ARCH-002, ARCH-003 |
| Phase 04 — bounded reasoning | REFL-001, REFL-002, SUBSYS-002, SUBSYS-003 |
| Phase 05 — memory/graph | SUBSYS-001, SUBSYS-004, MEM-001, MEM-002, RETR-001 |
| Phase 06 — diagnostics | DIAG-001 |
| Phase 07 — benchmarks | BENCH-001, BENCH-002, BENCH-003 |
| Phase 08 — tooling/docs | DOC-001 |
