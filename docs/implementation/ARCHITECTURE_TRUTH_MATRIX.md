# Architecture Truth Matrix

Baseline audit: `dev` at `8e45914e1b4b0ce66236be3b4c041001fb1fc5e8` on 2026-08-02. Current-state entries are updated through Phase 02.

`C:\PRIMA_integrated\Draft.png` is the target architecture, not the current call graph. The current code has multiple execution boundaries and no single end-to-end PRIMA path. In this document, “canonical production path” means the workflow path entered through `PrimaRuntime.process()` / `process_async()`; this is a baseline label, not an endorsement of the split.

## State definitions

| State | Meaning |
|---|---|
| active in canonical production path | Called by the current `process` workflow path. |
| active only in another production path | Called by a public runtime path that bypasses the workflow. |
| benchmark-only | Called by benchmark/evaluation code, not the canonical runtime path. |
| test-only | Reachable only from tests. |
| implemented but disconnected | Implementation exists but no current production call path reaches it. |
| missing | No implementation matching the target responsibility exists. |
| target architecture only | Shown or required by the target but not represented by the current runtime graph. |

## Matrix

| Component or responsibility | Current state | Evidence | Target disposition |
|---|---|---|---|
| `PrimaRuntime.process/process_async` workflow | active in canonical production path | `runtime/prima_runtime.py:296-325` calls `workflow.run`, persists turn memory, builds a result. | Become a thin adapter to the one typed public boundary. |
| Affect update | active in canonical production path | `workflow/prima_workflow.py:25-35`. | Select by execution profile. |
| Dense, sparse and temporal retrieval | active in canonical production path | `runtime/prima_runtime.py:54-57`; `memory/retrieval/retrieval_controller.py:45-49`. | Keep behind workflow ownership. |
| Planning | active in canonical production path | `workflow/prima_workflow.py:68-88`. | Keep in bounded lifecycle. |
| Reflection evaluation | active in canonical production path | `workflow/prima_workflow.py:91-138`. | Add bounded correction/retrieve/replan transitions. |
| Action selection/execution | active in canonical production path | `workflow/prima_workflow.py:145-171`. | Execute the selected LLM/tool action and validate output. |
| Output controller | active in canonical production path | `workflow/prima_workflow.py:174-192` returns `context.user_input`. | Shape generated/validated output, not echo input. |
| Episodic turn admission | active in canonical production path | `runtime/prima_runtime.py:341-369`. | Commit through workflow lifecycle. |
| Phase lifecycle events | active in canonical production path | `workflow/prima_workflow.py` publishes phase events. | Extend to typed maintenance events. |
| QA reasoning and LLM synthesis | active only in another production path | `runtime/prima_runtime.py:90-154,163-242` calls `ReasoningController` and `LLMClient` without `workflow.run`. | Route `factual_qa` through the workflow. |
| Direct document ingestion | active only in another production path | `runtime/prima_runtime.py:156-161` directly adds a semantic note. | Route `document_ingestion` through the same boundary. |
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
| Explicit task/profile routing | active in canonical production path | `runtime/route_profiles.py` deterministically covers all 25 task/profile pairs. Workflow enforcement is deferred. | Make the selected plan own workflow execution. |
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
