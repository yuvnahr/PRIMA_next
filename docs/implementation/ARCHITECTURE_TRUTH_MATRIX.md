# Architecture Truth Matrix

Baseline audit: `dev` at `8e45914e1b4b0ce66236be3b4c041001fb1fc5e8` on 2026-08-02. Current-state entries are updated through Phase 03.

`C:\PRIMA_integrated\Draft.png` is the target architecture, not the current call graph. “Canonical production path” now means a typed `PrimaRuntime.execute()` request followed by one workflow-owned task/profile route. The diagram still contains later-phase blocks that remain disconnected.

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
| Dense, sparse and temporal retrieval | active in canonical production path | `runtime/prima_runtime.py:54-57`; `memory/retrieval/retrieval_controller.py:45-49`. | Keep behind workflow ownership. |
| Planning | active in canonical production path | `workflow/prima_workflow.py:68-88`. | Keep in bounded lifecycle. |
| Reflection evaluation | active in canonical production path | `workflow/prima_workflow.py:91-138`. | Add bounded correction/retrieve/replan transitions. |
| Action selection/execution | active in canonical production path | `workflow/prima_workflow.py:145-171`. | Execute the selected LLM/tool action and validate output. |
| Answer generation | active in canonical production path | `workflow.answer_generation.AnswerGenerationController` calls the injected `LLMClient` and returns `GenerationResult`. | Keep model invocation workflow-owned and typed. |
| Output controller | active in canonical production path | Shapes existing typed generation, ingestion, classification or action results; it has no input fallback. | Remain result shaping only. |
| Episodic turn admission | active in canonical production path | `MemoryCommitController` runs as the final conversation workflow phase. | Add maintenance-event emission later. |
| Phase lifecycle events | active in canonical production path | `workflow/prima_workflow.py` publishes phase events. | Extend to typed maintenance events. |
| QA evidence acquisition and LLM synthesis | active in canonical production path | Workflow routes `EVIDENCE_ACQUISITION` through `ReasoningController`, then `ANSWER_GENERATION`; `answer_question` only converts the typed response. | Add bounded reflect/retrieve/replan behavior in Phase 04. |
| Document ingestion | active in canonical production path | Workflow selects `DOCUMENT_INGESTION → OUTPUT`; the compatibility method delegates to `execute`. | Emit maintenance events later; never generate by default. |
| HotpotQA execution | benchmark-only | `benchmarks/hotpotqa/experiment.py:136-174` combines direct ingestion with separate QA. | Thin adapter to typed runtime requests. |
| LoCoMo execution | benchmark-only | `benchmarks/locomo/experiment.py:39-153` combines workflow turns with separate QA. | Thin adapter to typed runtime requests. |
| GoEmotions systems | benchmark-only | `benchmarks/goemotions/experiment.py:38-105`; `systems.py` directly invokes classifier/LLM/affect variants. | Use `emotion_classification` with `affect_only`. |
| Graph traversal retrieval | benchmark-only | Instantiated by evaluation runners; absent from the runtime retrieval defaults. | Selectively enable through profile/configuration. |
| Graph reasoning engine | implemented but disconnected | Reached by `MemoryEvolutionEngine`, which has no production caller. | Integrate only where route/profile requires it. |
| Predictive world model | implemented but disconnected | Workflow action reads `world_prediction` metadata, but no workflow phase produces it. | Add bounded world-simulation transition. |
| Uncertainty estimator/gate | implemented but disconnected | Workflow action reads `uncertainty` metadata, but no workflow phase produces it. | Gate execute versus reflect/retrieve/replan. |
| Consolidation, abstraction and forgetting | implemented but disconnected | Maintenance/evolution engines are reached by validation scripts/tests, not production runtime. | Consume emitted maintenance events off the hot path. |
| Maintenance event subscribers | missing | Repository search finds publishers/helpers and tests but no production maintenance subscriber. | Add explicit cold-path consumers. |
| Procedural memory | missing | `memory/memory_types.py:8-34` has working, episodic, semantic and emotional only. | Add only when a phase defines storage/retrieval semantics. |
| Typed task request/result contract | active in canonical production path | `runtime/contracts.py`; `PrimaRuntime.execute` returns `PrimaResponse`. Legacy result types remain during migration. | Migrate every compatibility caller to the typed boundary. |
| Explicit task/profile routing | active in canonical production path | `runtime/route_profiles.py` validates all 25 pairs; `workflow.task_router` enforces the nine valid ordered routes. | Keep route plans synchronized with component diagnostics. |
| Input parser boundary | active in canonical production path | `PrimaRequest` validates versioned, extra-forbidding input before route selection. | Move any task-specific parsing behind workflow ownership. |
| Bounded correction loop | target architecture only | Reflection records a decision but does not transition back to retrieval/planning. | Bound retries and expose them in diagnostics. |
| Shared state/memory taxonomy in diagram | target architecture only | Current stores and runtime state do not implement the diagram’s complete taxonomy/links. | Introduce incrementally with evidence-backed activation. |
| Cross-encoder reranking by default | implemented but disconnected | `memory/retrieval/reranker.py:35-100` defaults to lexical scoring unless cross-encoder is enabled and loads. | Make actual strategy explicit in diagnostics. |

## Verified facts versus hypotheses

All matrix state assignments are verified from call sites and repository search. The following remain hypotheses and must not be represented as measured facts:

- Graph traversal, world simulation, affect adaptation, reflection or maintenance will improve benchmark scores.
- A cross-encoder should be the default reranker.
- The complete diagram should run for every task; short task-appropriate routes are the target.
- Existing benchmark scores measure the complete PRIMA wrapper or isolate Qwen improvement.
