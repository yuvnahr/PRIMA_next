# Embedding Pipeline Validation

Status: **FAIL — backend selection is centralized, but the full production path is not backend-safe across persisted data and does not apply identity/semantic representation.**

## Full execution path

### Runtime write path

```text
PrimaRuntime.process_async()
  -> PrimaWorkflow.run()
  -> MemoryImportanceEngine.decide()
       -> novelty_score()
       -> repository.query(stable_embedding(query))
  -> PrimaRuntime._persist_turn_memory()
       -> MemoryNote.create(content=user_input)
            -> stable_embedding(content)
                 -> embed_text(text, dimensions)
                      -> get_embedding_backend()
                           -> PRIMA_EMBEDDING_BACKEND / MODEL / DIMENSIONS
  -> MemoryRepository.add(note)
       -> InMemoryMemoryRepository: stores supplied vector
       -> ChromaMemoryRepository: stores supplied vector and metadata
```

`stable_embedding` is currently a compatibility shim, not a hash-only bypass: `memory/memory_note.py:39-42` delegates to `memory.embedding_backend.embed_text`, and `embed_text` selects the configured backend in `memory/embedding_backend.py:134-137`.

### Runtime read path

```text
PrimaRuntime.answer_question()
  -> RetrievalRequest(query=question)
  -> RetrievalController.retrieve(request)
  -> DenseRetrievalStrategy.retrieve(request, repository)
  -> RetrievalRequest.embedding()
       -> stable_embedding(request.query)
            -> embed_text()
                 -> configured EmbeddingBackend.embed()
  -> repository.query(query_vector)
  -> dense results / LoCoMo consumer
```

The workflow retrieval path in `workflow/prima_workflow.py:53-65` follows the same `RetrievalRequest -> RetrievalController` route.

### Event-memory path

```text
EventMemoryBuilder.build(segment)
  -> EventMemory.to_memory_note()
       -> stable_embedding(event.embedding_text)
            -> configured backend
       -> MemoryNote.create(..., embedding=vector)
  -> repository.add(note)
```

Event memories therefore use the configured backend, but bypass the default `MemoryNote.create` embedding call by supplying a vector explicitly. The event embedding text is structured in `memory/event_memory/event.py:39-50`; it is not produced by `build_semantic_representation` or identity normalization.

### Evolution and reflection paths

```text
MemoryEvolutionEngine.evolve()
  -> MemoryNote.create(summary, embedding=stable_embedding(summary, ...))
  -> repository.add()

ReflectionMemory.to_memory_note() / Rule.to_memory_note()
  -> MemoryNote.create(..., embedding=stable_embedding(text))
  -> reflection repository -> memory repository
```

These paths also reach the configured backend through the shim, but do not apply the production semantic-representation or identity-normalization stages.

## Call graph

```text
Conversation / runtime input
  -> PrimaRuntime._persist_turn_memory
  -> MemoryNote.create
  -> stable_embedding [compatibility API]
  -> embed_text
  -> get_embedding_backend
  -> EmbeddingBackend.embed
  -> MemoryRepository.add/update

Question
  -> RetrievalRequest
  -> RetrievalRequest.embedding
  -> stable_embedding
  -> embed_text
  -> get_embedding_backend
  -> EmbeddingBackend.embed
  -> DenseRetrievalStrategy
  -> MemoryRepository.query
  -> LoCoMo/runtime consumer
```

## Detected legacy paths

| Location | Finding | Classification |
|---|---|---|
| `memory/memory_note.py:39-42` | `stable_embedding()` remains the public name used by callers | Compatibility shim; no backend bypass today |
| `memory/retrieval/retrieval_request.py:26` | Query embedding calls the shim | Production path; configured backend is reached |
| `memory/maintenance/memory_importance.py:130` | Novelty scoring calls the shim | Production path; configured backend is reached |
| `memory/evolution/memory_evolution_engine.py:59` | Evolved notes call the shim explicitly | Production path; configured backend is reached |
| `memory/event_memory/event.py:43` | Event notes call the shim explicitly | Production path; configured backend is reached |
| `reflection/reflection_memory.py:34,90` | Reflection/rule notes call the shim explicitly | Production path; configured backend is reached |
| `evaluation/runners/retrieval_optimization_runner.py:16,193` | Evaluation diagnostics import/call the shim directly | Evaluation-only legacy path |
| `benchmarks/locomo/phase10_semantic_representation.py:278,326` | Benchmark embedding and query embedding call the shim directly | Benchmark-only path; excluded from production conclusion |

No production caller was found that directly invokes `stable_hash_embedding()` or recreates the hash algorithm. The similarly named private `affect/emotion_classifier.py:_stable_embedding` is an affect-classifier lexicon feature, not a memory/retrieval vector path.

## Duplicate embedding logic

1. `memory/embedding_backend.py:67-79` contains the stable hash implementation, while `memory/embedding_backend.py:140-143` exposes a second constructor-style stable baseline API. This is intentional baseline/test compatibility, but it is a second public surface.
2. `memory/memory_note.py:39-42` adds another public compatibility surface over `embed_text`.
3. `memory/event_memory/event.py`, `memory/evolution/memory_evolution_engine.py`, and `reflection/reflection_memory.py` duplicate the pattern “compute vector, then pass `embedding=` to `MemoryNote.create`”. This is not a second backend algorithm, but it makes the embedding policy non-central.
4. `memory/semantic_representation.py` and `memory/event_memory/event_builder.py` each implement overlapping structured extraction helpers (`_summary`, capitalized terms, term matching, locations, and object extraction). They are representation logic rather than embedding algorithms, but they create two incompatible pre-embedding representations.
5. The vendored LoCoMo external code contains independent OpenAI/Hugging Face embedding utilities. It is outside PRIMA production retrieval and is not used by the audited runtime path.

## Evaluation-only embedding path

`benchmarks/locomo/phase10_semantic_representation.py` is explicitly an ablation/benchmark path. It performs:

```text
raw turn -> optional build_semantic_representation(..., normalize_identities=True)
          -> stable_embedding(embedded_text)
          -> repository.add
query -> stable_embedding(query) -> repository.query
```

This proves the representation and identity features in an evaluation harness, not in `Conversation -> MemoryNote.create`. `evaluation/runners/retrieval_semantic_runner.py` and `retrieval_benchmark_runner.py` use `MemoryNote.create`, but only with raw `memory_text`; they do not constitute production paths.

## Stale-vector and backend-change validation

**Failed.** The backend configuration includes backend name, model identifier, and dimensions, but no backend fingerprint is stored with each note or repository collection. `ChromaMemoryRepository` creates/reuses fixed collections at `memory/memory_repository.py:89-95`, and `add/update` stores the supplied vector without checking provenance at `:97-115`.

On a later process using a different `PRIMA_EMBEDDING_BACKEND`, existing Chroma vectors are loaded unchanged by `MemoryNote.from_record` and queried against newly generated vectors. There is no repository startup check, collection version, migration, or rebuild operation. `MemoryNote.from_record` also has an eager fallback expression at `memory/memory_note.py:208`: the fallback embedding is evaluated even when `record["embedding"]` exists, although the stored vector still wins.

The in-memory repository is process-local, so it does not retain vectors across backend changes; this masks the persistent-repository failure in tests.

## Identity normalization before embedding

**Failed for production.** `build_semantic_representation(..., normalize_identities=True)` applies `IdentityNormalizer.observe` before serialization (`memory/semantic_representation.py:92-96`), but no runtime `MemoryNote.create` caller invokes it. Production note content is embedded as raw input, and event memory uses raw/structured event text without the normalizer.

## Semantic representation before embedding

**Failed for production.** `build_semantic_representation` is referenced by `benchmarks/locomo/phase10_semantic_representation.py` only. `MemoryNote.create` embeds `content` directly when no explicit vector is supplied (`memory/memory_note.py:181-190`). Production conversation, event, reflection, and evolution paths do not pass a serialized semantic representation into the backend.

## Dead code

- `memory/semantic_representation.py` is not on the production conversation or retrieval call graph; it is currently benchmark support code.
- `memory/identity_normalization.py` is not on the production embedding call graph; its public helper is only used by the semantic-representation benchmark flow.
- `stable_hash_embedding()` is retained as a frozen baseline/test API and is not a production retrieval path.
- The external LoCoMo embedding helpers are vendored benchmark dependencies, not PRIMA production code.

## Recommendations

1. Add a backend fingerprint to stored note metadata and Chroma collection metadata. On repository open, detect a mismatch and rebuild every stored vector from the canonical embedding text before serving retrieval. Do not silently mix dimensions or model versions.
2. Make one production embedding entry point own both pre-processing and backend selection. At minimum, route conversation and event text through identity normalization and semantic serialization before `embed_text`; preserve the original content separately for display.
3. Remove direct production imports of the compatibility-named `stable_embedding` after the central entry point exists. Keep the shim only for compatibility/tests, or rename it to make its configured-backend behavior explicit.
4. Add one integration check that selects two backend configurations, writes/loads through the persistent repository, and asserts that a backend change invalidates or rebuilds stored vectors. Existing backend unit tests validate provider selection, not persisted-vector invalidation.

## Scope and exclusions

No retrieval strategy, ranking logic, benchmark implementation, or evaluation result was changed. The only artifact produced by this audit is this report.
