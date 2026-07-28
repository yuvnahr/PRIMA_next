# Reasoning runtime audit

Audit date: 2026-07-22. Scope: production runtime only; benchmark packages were not changed.

## Current call paths

The public factual-QA entry point is `runtime/prima_runtime.py:PrimaRuntime.answer_question`.

Old path:

`PrimaRuntime.answer_question` → `RetrievalController.retrieve(RetrievalRequest)` → `RuntimeContextBuilder.build` → `LLMClient.chat` (with extractive fallback) → `RuntimeResult`.

New path:

`PrimaRuntime.answer_question` → `ReasoningController.answer` → unchanged `RetrievalController.retrieve` for each hop → `EvidenceIntegrator` / `SufficiencyVerifier` / `StoppingPolicy` → existing `RuntimeContextBuilder` and `LLMClient` synthesis path → `AnswerResult`.

## Existing interfaces retained

- `memory/retrieval/retrieval_controller.py:RetrievalController.retrieve` remains the only retrieval gateway. Dense, sparse, temporal, fusion, reranking, query expansion, confidence, and embeddings are unchanged.
- `runtime/context_builder.py:RuntimeContextBuilder` remains the context-budget boundary.
- `llm/llm_client.py:LLMClient` remains the provider abstraction.
- `reflection/reflection_engine.py`, `planning/`, and `events/` are not invoked by default: the existing runtime has no stable synchronous QA event hook. No second event bus was added.
- `runtime/prima_runtime.py:process` remains the interactive workflow and its memory-write path. Reasoning state is not passed to it.

## Boundaries and changed files

The controller is request-local (`EvidenceState`) and never writes memory. `ReasoningRequest.session_id` is trace-only scope metadata; no state is shared by session or request. `reasoning/*` imports only production retrieval models and does not import `benchmarks/*`.

Modified production integration: `runtime/prima_runtime.py`. New production package: `reasoning/*`. New focused checks: `tests/reasoning/test_controller.py`.
