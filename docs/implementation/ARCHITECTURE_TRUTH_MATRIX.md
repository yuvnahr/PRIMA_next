# Architecture Truth Matrix

Baseline audit: `dev` at `8e45914e1b4b0ce66236be3b4c041001fb1fc5e8` on 2026-08-02. Current-state entries are updated through Phase 15.

The target diagram is an architectural intent, not evidence of the current call graph. “Canonical production path” means a typed `PrimaRuntime.execute()` request followed by one workflow-owned task/profile route. The executable route diagnostics and the matrix below are the status labels of record.

## State definitions

| State | Meaning |
|---|---|
| active in canonical production path | Called through `PrimaRuntime.execute()` and its selected workflow route. |
| active only in another production path | Called by production code outside the canonical runtime/workflow lifecycle. |
| benchmark-only | Called by benchmark/evaluation code, not the canonical runtime path. |
| test-only | Reachable only from tests. |
| implemented but disconnected | Implementation exists but no current production call path reaches it. |
| missing | No implementation matching the target responsibility exists. |
| target architecture only | Shown or required by the target but not represented by the current runtime graph. |

## Matrix

| Component or responsibility | Current state | Evidence | Target disposition |
|---|---|---|---|
| `PrimaRuntime.process/process_async` workflow | active in canonical production path | Both methods construct a typed conversation request and delegate to `execute`; only `execute` calls `workflow.run`. | Keep as thin compatibility adapters. |
| Affect update | active in canonical production path | `workflow/prima_workflow.py:25-35`. | Select by execution profile. |
| Profile-driven hybrid retrieval | active in canonical production path | `RetrievalController` selects no retrieval for `model_only`, dense+sparse for `simple_rag`, and dense+sparse+temporal+graph for capable `prima_full` routes. | Keep task/profile selection explicit in diagnostics. |
| Memory index facade | active in canonical production path | Runtime ingestion, admitted memory writes and retrieval use `MemoryIndex`; content remains solely in the configured memory repository while graph records retain IDs and derived metadata. | Keep ChromaDB as the production content source of truth. |
| Query rewriting | active in canonical production path | `QueryRewrite` records the original query, typed analysis, expanded query and deterministic provenance before strategy execution. | Keep rewriting benchmark-independent and traceable. |
| Context compression | active in canonical production path | `ContextCompressor` applies the request token budget after evidence acquisition, preserves evidence/source mappings and reports loss/drops; ablation is explicit. | Replace approximate word tokens only if a configured model tokenizer is required. |
| Planning | active in canonical production path | `workflow/prima_workflow.py:68-88`. | Keep in bounded lifecycle. |
| Reflection evaluation | active in canonical production path | Workflow reflection emits typed advice; the orchestration engine accepts or rejects it under explicit budgets and routes corrections without direct subsystem coupling. | Calibrate advice confidence and utility using production observations. |
| Action selection/execution | active in canonical production path | `workflow/prima_workflow.py:145-171`. | Execute the selected LLM/tool action and validate output. |
| Answer generation | active in canonical production path | `workflow.answer_generation.AnswerGenerationController` calls the injected `LLMClient` and returns `GenerationResult`. | Keep model invocation workflow-owned and typed. |
| Output controller | active in canonical production path | Shapes existing typed generation, ingestion, classification or action results; it has no input fallback. | Remain result shaping only. |
| Episodic turn admission | active in canonical production path | `MemoryCommitController` runs as the final conversation workflow phase. | Add maintenance-event emission later. |
| Versioned cognitive state ownership | active in canonical production path | Every canonical task route begins with `STATE_LOAD`; task-appropriate successful/abstained routes finish with optimistic `STATE_COMMIT`. | Keep state transitions workflow-owned; add bounded reasoning deltas later. |
| Runtime/storage mode policy | active in canonical production path | `RuntimeMode` plus `select_memory_repository` exposes test/benchmark/production selection and rejects ephemeral production storage. | Production source of truth remains configured ChromaDB; no fallback. |
| Phase lifecycle events | active in canonical production path | `workflow/prima_workflow.py` publishes phase events. | Extend to typed maintenance events. |
| QA evidence acquisition and LLM synthesis | active in canonical production path | Workflow routes `EVIDENCE_ACQUISITION` through `ReasoningController`, then `ANSWER_GENERATION`; `answer_question` only converts the typed response. | Add bounded reflect/retrieve/replan behavior in a future bounded-reasoning phase. |
| Document ingestion | active in canonical production path | Workflow selects `DOCUMENT_INGESTION → OUTPUT`; the compatibility method delegates to `execute`. | Emit maintenance events later; never generate by default. |
| HotpotQA execution | active in canonical production path | Benchmark adapters submit document ingestion and factual-QA `PrimaRequest` objects through `PrimaRuntime.execute()` with declared profiles. | Keep benchmark-specific scoring outside the runtime. |
| LoCoMo execution | active in canonical production path | Conversation turns, controlled ingestion, and questions all use `PrimaRuntime.execute()`; paired modes select explicit admission and retrieval policies. | Keep core metrics independent of semantic extras. |
| GoEmotions systems | active in canonical production path | Every prediction uses `TaskKind.EMOTION_CLASSIFICATION` and `affect_only`; diagnostics prove QA retrieval, planning, world simulation, and tools are excluded. | Preserve component-benchmark attribution. |
| Graph traversal retrieval | active in canonical production path | Capable `prima_full` retrieval constructs `GraphTraversalStrategy`; diagnostics and spies prove invocation while cheaper profiles report it disabled. | Keep graph optional and profile-driven. |
| Graph reasoning engine | active in canonical production path | Graph traversal invokes centrality reasoning and records the contribution in result explanations and runtime diagnostics. | Do not claim learned graph reasoning. |
| Predictive world model | active in canonical production path | `WORLD_SIMULATION` invokes the existing deterministic symbolic `StateSimulator` in `prima_full`; its typed result reaches the action context and diagnostics. | Learned prediction remains outside the architecture claim. |
| Uncertainty estimator/gate | active in canonical production path | `UNCERTAINTY_ESTIMATION` aggregates available retrieval/planner/affect/state/policy/reflection signals; `EXECUTION_DECISION` applies explicit thresholds before execution. | Calibration against production observations remains future work. |
| Consolidation, abstraction and forgetting | active in canonical production path | Admitted memory enqueues typed work for the bounded maintenance supervisor, whose stages cover salience, consolidation, evolution/abstraction, graph refresh, decay, and soft forgetting. | Keep this cold path observable and bounded. |
| Maintenance event subscribers | active in canonical production path | The local supervisor consumes typed maintenance events with retry, duplicate suppression, cancellation, restart, and durable failure reporting. | External brokers remain out of scope. |
| Procedural memory | active in canonical production path | `MemoryType.PROCEDURAL` admits only verified successful tool executions or explicit imports; tool planning retrieves it and records used procedure IDs. | Add richer procedure schemas only when real tool contracts require them. |
| Typed task request/result contract | active in canonical production path | `runtime/contracts.py`; `PrimaRuntime.execute` returns `PrimaResponse`. Legacy result types remain during migration. | Migrate every compatibility caller to the typed boundary. |
| Explicit task/profile routing | active in canonical production path | `runtime/route_profiles.py` validates all 25 pairs; `workflow.task_router` enforces the nine valid ordered routes. | Keep route plans synchronized with component diagnostics. |
| Input parser boundary | active in canonical production path | `PrimaRequest` validates versioned, extra-forbidding input before route selection. | Move any task-specific parsing behind workflow ownership. |
| Bounded correction loop | active in canonical production path | Accepted advice routes to retrieval, replanning, action retry or regeneration; every attempt records real before/after snapshots and explicit budget decisions. | Later phases may add richer validators without changing ownership. |
| Shared state/memory taxonomy in diagram | target architecture only | Cognitive and five memory types are owned, but maintenance scheduling and several complete diagram links remain absent. | Introduce remaining blocks incrementally with evidence-backed activation. |
| Reranker backend selection | active in canonical production path | Lexical fallback and cross-encoder have distinct typed identities; requested/active backend is diagnostic, mismatch fails, and required cross-encoder load failure stops preflight. | Keep benchmark manifests pinned to the reported backend. |

## Verified facts versus hypotheses

All matrix state assignments are verified from call sites and repository search. The following remain hypotheses and must not be represented as measured facts:

- Graph traversal, world simulation, affect adaptation, reflection or maintenance will improve benchmark scores.
- A cross-encoder should be the default reranker.
- The complete diagram should run for every task; short task-appropriate routes are the target.
- Existing benchmark scores measure the complete PRIMA wrapper or isolate Qwen improvement.
