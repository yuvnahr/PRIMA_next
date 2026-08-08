# Phase 07 — Retrieval, Graph, Reranking and Context Alignment

## Outcome

The canonical workflow now owns one profile-driven retrieval lifecycle. `MemoryIndex` coordinates the configured repository with derived graph, temporal and metadata access without copying memory content into a competing store. Full-profile retrieval invokes dense, sparse, temporal and graph strategies when available; simple RAG invokes dense and sparse only; model-only routes do not retrieve.

## Contract and behavior changes

- `MemoryIndex` is the lifecycle facade for repository writes/queries and derived graph indexing. Chroma remains the production content source of truth.
- `QueryRewrite` records original query, entity/relation/time analysis, deterministic expansion and provenance.
- `GraphTraversalStrategy` invokes the existing deterministic `GraphReasoningEngine` centrality calculation; diagnostics prove calls and explicit skips.
- `RerankerBackend` distinguishes `disabled`, `lexical_fallback` and `cross_encoder`. Required cross-encoder load failures and request/backend mismatches fail instead of silently changing backend.
- `ContextCompressor` enforces the evidence token budget, preserves evidence/source citation mapping, reports dropped evidence and loss, and supports explicit no-compression ablation.
- `MemoryType.PROCEDURAL` and `MemoryLevel.VERIFIED_PROCEDURE` are admitted only for verified successful tool executions or explicit imports.
- Tool requests retrieve procedural memory before planning; the planner records consumed procedural IDs. Failed, blocked, abstained and internal no-tool executions cannot create procedural memory.
- Runtime diagnostics report actual retrieval strategies, graph reasoning, index capabilities, reranker backend and compression audit data rather than inferring activation from class existence.

## Changed files

- Memory/index lifecycle: `memory/memory_index.py`, `memory/memory_types.py`, `memory/graph/graph_builder.py`, `memory/__init__.py`
- Retrieval: `memory/retrieval/query_analysis.py`, `retrieval_request.py`, `retrieval_controller.py`, `graph_strategy.py`, `hybrid_fusion.py`, `reranker.py`, `context_compressor.py`
- Configuration: `.env.example`
- Workflow/planning/runtime: `workflow/prima_workflow.py`, `workflow/task_router.py`, `workflow/execution_context.py`, `planning/action_selector.py`, `runtime/route_profiles.py`, `runtime/prima_runtime.py`
- Tests: `tests/retrieval/test_phase07_retrieval_alignment.py`
- Ledger: `docs/implementation/ARCHITECTURE_TRUTH_MATRIX.md`, `docs/implementation/STATUS.md`

## Verification

```text
Phase 07 focused behavior tests:
venv\Scripts\python.exe -m pytest -q tests\retrieval\test_phase07_retrieval_alignment.py
5 passed in 13.48s.

Retrieval, graph, planning, runtime and state regression selection:
venv\Scripts\python.exe -m pytest -q tests\retrieval\test_phase07_retrieval_alignment.py tests\retrieval\test_retrieval_phase2.py tests\retrieval\test_retrieval_router.py tests\retrieval\test_graph_diagnostics.py tests\graph\test_graph_phase3.py tests\planning\test_planning_phase2.py tests\runtime\test_canonical_contract.py tests\state\test_state_ownership.py tests\integration\test_runtime_pipeline.py
36 passed in 20.69s.

Focused runtime diagnostics rerun after correcting the test embedding fixture:
venv\Scripts\python.exe -m pytest -q tests\retrieval\test_phase07_retrieval_alignment.py::test_runtime_diagnostics_report_actual_profile_capabilities
1 passed in 7.84s.

Expanded Phase 07 suite after adding explicit-import and admission enforcement:
venv\Scripts\python.exe -m pytest -q tests\retrieval\test_phase07_retrieval_alignment.py
7 passed in 8.64s.

Focused Ruff:
venv\Scripts\python.exe -m ruff check memory\memory_types.py memory\memory_index.py memory\graph\graph_builder.py memory\retrieval\query_analysis.py memory\retrieval\retrieval_request.py memory\retrieval\reranker.py memory\retrieval\graph_strategy.py memory\retrieval\context_compressor.py memory\retrieval\retrieval_controller.py memory\retrieval\hybrid_fusion.py planning\action_selector.py workflow\execution_context.py workflow\prima_workflow.py workflow\task_router.py runtime\route_profiles.py runtime\prima_runtime.py tests\retrieval\test_phase07_retrieval_alignment.py
All checks passed.

Focused mypy:
venv\Scripts\python.exe -m mypy memory\memory_index.py memory\retrieval\context_compressor.py memory\retrieval\reranker.py memory\retrieval\retrieval_controller.py workflow\prima_workflow.py runtime\prima_runtime.py
Success: no issues found in 6 source files. One informational unused override note was emitted for `benchmarks.preflight`.

Consolidated Phase 03/04/06/07 runtime regression selection:
venv\Scripts\python.exe -m pytest -q tests\retrieval\test_phase07_retrieval_alignment.py tests\runtime\test_canonical_contract.py tests\integration\test_runtime_pipeline.py tests\state\test_state_ownership.py tests\workflow\test_phase06_correction_loop.py
33 passed in 12.55s.

Final complete CI:
venv\Scripts\python.exe -m compileall -q -x "(^|[\\/])(\.git|\.venv|venv|external)([\\/]|$)" .
Passed with no output.

venv\Scripts\python.exe -m pytest -q
253 passed, 1 skipped in 126.06s. The skip is the existing optional/environment-gated test; Phase 07 adds no skip.

venv\Scripts\python.exe -m ruff check .
All checks passed.

venv\Scripts\python.exe -m mypy .
Success: no issues found in 213 source files.

venv\Scripts\python.exe -m bandit -q -c bandit.yaml -r .
Passed. Bandit emitted one existing informational `nosec encountered (B603)` warning for `benchmarks/hotpotqa/experiment.py:66` and no failure.

venv\Scripts\python.exe scripts\run_xenon.py
Passed against the allowlisted PRIMA production roots only.
```

## Risks and deferred work

- Graph centrality and query rewriting are deterministic symbolic/lexical operations, not learned reasoning claims.
- Initial graph synchronization is derived from repository metadata and can be optimized incrementally if production profiling demonstrates startup pressure.
- Context token counts use deterministic whitespace tokens; a provider-specific tokenizer remains deferred until a configured model contract requires it.
- Maintenance scheduling, graph deletion/stale-edge cleanup and richer procedural schemas remain later work.
- The pre-existing dirty `benchmarks/locomo-v2/external` submodule remains untouched.
- `C:\PRIMA_integrated\Draft.png` remains the target architecture; disconnected maintenance blocks are not reported active.

Stop here. Do not begin the next phase.
