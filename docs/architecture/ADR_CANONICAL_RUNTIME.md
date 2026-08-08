# ADR: Canonical Runtime Contract and Route Profiles

- Status: Accepted
- Date: 2026-08-02
- Decision owners: PRIMA runtime and workflow maintainers
- Scope: Public contract and deterministic route selection

## Context

`C:\PRIMA_integrated\Draft.png` is the target architecture, not the current call graph. The Phase 00 audit proved that conversation turns, factual QA, document ingestion, and emotion classification currently cross different boundaries. A single typed boundary is required before internal orchestration can be migrated safely.

## Decision

The only target public execution boundary is:

```text
PrimaRuntime.execute(PrimaRequest) -> await PrimaResponse
```

`PrimaRequest` carries a `TaskKind` and `ExecutionProfile`. `runtime.route_profiles.select_route` deterministically selects an ordered `RoutePlan` before workflow execution. Production route selection contains no benchmark names or imports.

The synchronous compatibility method is `PrimaRuntime.execute_sync(request)`. It may call `asyncio.run` only when no event loop is active. If a loop is active it raises `RuntimeError` directing the caller to `await PrimaRuntime.execute(request)`.

Existing `process`, `process_async`, `answer_question`, and `ingest_document` methods remain operational during migration. They are compatibility surfaces, not alternate target architectures.

## Contract models

All public models live in `runtime/contracts.py` and reject unknown fields.

| Model | Responsibility |
|---|---|
| `PrimaRequest` | Versioned request ID, task, profile, input, session and caller metadata. |
| `PrimaResponse` | Versioned status, output, state delta, evidence, diagnostics and errors. |
| `TaskKind` | Conversation, factual QA, document ingestion, emotion classification or tool request. |
| `ExecutionProfile` | Model only, simple RAG, full PRIMA, affect only or ingestion only. |
| `ExecutionStatus` | Completed, not implemented, rejected, failed or cancelled lifecycle state. |
| `ExecutionOutcome` | Answered, abstained, failed, ingested, classified or cancelled semantic result. |
| `ComponentCapability` | Whether a component is callable through the canonical boundary and why. |
| `StateDelta` | Serializable committed state changes. |
| `EvidenceReference` | Runtime evidence only; never benchmark gold/supporting-fact data. |
| `RuntimeDiagnostics` | Route name plus planned, executed, skipped and available components. |

Request, response and diagnostics JSON use schema version `1.0`. Route selection does not prove execution. Only `executed_components` records calls made by the canonical boundary; `planned_components` records target intent and `skipped_components` records every component not actually called, including planned capabilities deferred in this phase.

## Task/profile matrix

Every one of the 25 combinations has a deterministic decision. `invalid` combinations are rejected with the valid profiles named in the error.

| Task kind | `model_only` | `simple_rag` | `prima_full` | `affect_only` | `ingestion_only` |
|---|---|---|---|---|---|
| `conversation` | valid | valid | valid | invalid | invalid |
| `factual_qa` | valid | valid | valid | invalid | invalid |
| `document_ingestion` | invalid | invalid | invalid | invalid | valid |
| `emotion_classification` | invalid | invalid | invalid | valid | invalid |
| `tool_request` | invalid | invalid | valid | invalid | invalid |

### Profile intent

| Profile | Selected lifecycle |
|---|---|
| `model_only` | Parse/control/state → model execution → validation/output → state/memory commit → maintenance event. |
| `simple_rag` | Parse/control/state → rewrite/dense/sparse/temporal/fusion/rerank/confidence/compress → model → validation/output/commit/events. |
| `prima_full` | Parse/control/state → affect → hybrid plus graph retrieval → plan/world/uncertainty/reflection → model → validation/output/commit/events. |
| `affect_only` | Parse/control/state → emotion classifier/affect → state commit/output. No QA retrieval, planning, world, reflection or model generation. |
| `ingestion_only` | Parse/control → document encoder/index/memory commit → state commit/events/output acknowledgement. No answer generation. |

`tool_request/prima_full` replaces model execution with policy-gated tool execution. Exact ordered component tuples in `runtime/route_profiles.py` are the executable source of truth.

## Short-route rules

- GoEmotions adapters must submit `TaskKind.EMOTION_CLASSIFICATION` with `ExecutionProfile.AFFECT_ONLY`. They must not activate query rewriting, QA retrieval, planning, world simulation, uncertainty, reflection, model answering or tool execution merely to claim architectural unity.
- Document ingestion must submit `TaskKind.DOCUMENT_INGESTION` with `ExecutionProfile.INGESTION_ONLY`. It may encode/index/commit memory and emit maintenance events, but `PrimaResponse.output_text` remains `None`; no answer model is invoked.
- Every component not actually called appears in `RuntimeDiagnostics.skipped_components`.
- Benchmark packages may construct requests but production runtime packages never import benchmark modules.

## Current migration state

All nine valid task/profile combinations enter a workflow-owned route. Conversation and factual QA use injected answer generation; ingestion validates/indexes without generation; affect-only classification skips QA phases. Full profiles invoke graph retrieval, symbolic world simulation, uncertainty decisions, and bounded correction. Routes that admit memory finish with a non-blocking maintenance-enqueue phase; a local bounded supervisor performs cold-path work and exposes calls, skips, queue state, retries, and failures in diagnostics.

## Phase 04 state and memory ownership

Every canonical route now loads versioned session state through an injected `StateManager` and commits task-appropriate successful/abstained state with an expected-version check. Conversation and factual QA use the same session key. Runtime mode is explicit in diagnostics: tests may use deterministic memory, benchmarks declare their repository in runtime/run manifests, and production preflight requires persistent ChromaDB plus persistent JSON cognitive state. Failed/cancelled responses do not commit cognitive state; memory admission decisions are explicit for every task outcome.

## Final ownership of the target diagram

The following table assigns every named block in `Draft.png`. “Final owner” is the intended package/layer, not proof that the block is currently connected.

| Diagram block | Final owner |
|---|---|
| User Input | External caller; represented only by `runtime.contracts.PrimaRequest`. |
| Input Parser | `runtime` contract validation and a workflow-invoked parser boundary. |
| Cognitive Control Bus | `workflow.PrimaWorkflow`; owns the complete request lifecycle. |
| Task Scheduler | `workflow.orchestration_engine`; bounded phase scheduling/cancellation. |
| State Manager | `workflow` coordinates; `state` owns state models; repositories own persistence. |
| Policy Router | `runtime.route_profiles` selects task/profile intent; `workflow.task_router` executes it. |
| Execution Dispatcher | `workflow.controller_registry` and orchestration engine; no subsystem self-dispatch. |
| Persistent Cognitive State | `state.cognitive_state`/`state.emotional_state`, accessed through workflow context. |
| Goal State | `state` representation; `planning.goal_selector` proposes changes. |
| Emotional State | `state.emotional_state`; `affect` emits typed updates. |
| Task State | `workflow.execution_context` and planning plan status. |
| Confidence State | `uncertainty` owns confidence state/decisions; workflow commits deltas. |
| Environment State | `state` model populated only by policy-gated action/tool observations. |
| Dynamic Affect Engine | `affect.DynamicAffectEngine`; never orchestrates other packages. |
| Expanded Hybrid Retrieval | `memory.retrieval.RetrievalController`, invoked only by workflow. |
| Query Rewriter | `memory.retrieval.query_analysis`/resource-backed expansion boundary. |
| Dense Retrieval | `memory.retrieval.dense_strategy`. |
| Sparse Retrieval | `memory.retrieval.sparse_strategy`. |
| Temporal Retrieval | `memory.retrieval.temporal_strategy`. |
| Graph Traversal | `memory.retrieval.graph_strategy` over the memory graph repository. |
| Fusion Layer | `memory.retrieval.hybrid_fusion`. |
| Cross Encoder Reranker | `memory.retrieval.reranker`; actual/fallback strategy reported in diagnostics. |
| Confidence Estimator | Retrieval confidence remains in `memory.retrieval`; execution uncertainty remains in `uncertainty`. |
| Context Compressor | Retrieval/context shaping boundary owned by `memory.retrieval`, consumed by `runtime.context_builder`. |
| Graph Reasoning Engine | `memory.graph.graph_reasoning_engine`, selected only by profiles that require it. |
| Memory Index Layer | `memory.memory_repository` plus vector/graph/temporal/metadata index adapters. |
| Vector Index | Memory repository embedding/vector adapter. |
| Graph Index | `memory.graph` repository. |
| Temporal Index | Memory repository temporal metadata/index adapter. |
| Metadata Index | Memory repository metadata adapter. |
| Memory Taxonomy | `memory.memory_types` and typed stores. |
| Working Memory | `memory.stores` working store. |
| Episodic Memory | `memory.stores` episodic store. |
| Semantic Memory | `memory.stores` semantic store. |
| Procedural Memory | `memory.memory_types`; admitted only from verified successful tool procedures or explicit imports. |
| Emotional Memory | `memory.stores` emotional store fed by typed affect metadata. |
| Task Planner | `planning.TaskPlanner`, invoked only by workflow. |
| Goal Selection | `planning.goal_selector`. |
| Action Selection | `planning.action_selector`; selects intent but does not execute it. |
| Memory Integration | Planning context adapter consumes workflow-supplied memory; no direct retrieval call. |
| State Transition Planning | Planning models propose `StateDelta`; workflow owns commit. |
| World Model | `world` package, invoked as a bounded workflow stage. |
| Uncertainty Estimator | `uncertainty` package; decides execute versus bounded correction transitions. |
| Action / Execution Engine | `action.ActionExecutor`; delegates policy-gated tools to `tools`. |
| Final Output | `runtime.contracts.PrimaResponse`, shaped after workflow output validation. |
| Adaptive Reflection Pipeline | `reflection.AdaptiveReflectionPipeline`, invoked and bounded by workflow. |
| Trigger Evaluation | `reflection.reflection_engine`/trigger policy. |
| Constraint Check | `reflection` coordinates; planning/action policies supply constraints. |
| Logical Validation | `reflection.verifier_adapter` and output validation boundary. |
| Tool Verification | `tools.tool_validator` and action execution policy. |
| Hallucination Detection | `reflection.verifier_adapter` grounding checks; no benchmark gold inputs. |
| Correction Proposal | `reflection` emits typed advice/state; workflow decides the transition. |
| Re-evaluation | `reflection` evaluates corrected output; workflow enforces retry limits. |
| Event / Message Stream | `events.EventBus` and typed publishers/subscribers; asynchronous communication only. |
| Background Maintenance Pipeline | `memory.maintenance`/`memory.evolution`, triggered through events off the hot path. |
| Experience Stream | Typed committed runtime/memory events owned by `events`; maintenance consumes them. |
| Memory Encoder | `memory.embedding_pipeline`/configured embedding backend. |
| Salience Scoring | `memory.maintenance.salience_manager` and importance policy. |
| Short-Term Buffer | Bounded `MaintenancePipeline.short_term_buffer`, populated only by admitted-memory events. |
| Consolidation | `memory.maintenance.consolidation_engine`. |
| Semantic Abstraction | `memory.evolution.semantic_abstraction_engine`. |
| Long-Term Compression | Future maintenance policy under `memory.maintenance`; must be event-driven and tested. |
| Forgetting / Decay | `memory.maintenance.forgetting_policy`; soft retention changes, not hidden deletion. |

Arrow ownership is also fixed: synchronous/hot-path arrows are workflow calls; dashed cold-path arrows are event-bus delivery; dotted arrows are typed metadata/state deltas; bidirectional arrows are repository interfaces, never direct cross-package mutation.

## Consequences

- Callers have one stable typed target while legacy methods can be migrated incrementally.
- Invalid task/profile combinations fail deterministically before subsystem work.
- A class existing or a route selecting it never counts as execution; diagnostics must record calls.
- Dedicated long-term compression remains deferred; the implemented cold path currently covers encoding, salience, bounded buffering, consolidation, semantic abstraction/evolution, graph refresh, retention decay, and soft forgetting.
- Benchmark metrics remain unchanged in Phase 03; benchmark-specific adapter migration remains Phase 07.
