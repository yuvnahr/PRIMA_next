# PRIMA-NEXT Cognitive Subsystems

Python reimplementation of the PRIMA affect subsystem, PRIMA-NEXT hierarchical memory fabric, and PRIMA-NEXT adaptive reflection pillar. The packages preserve legacy AWARE/A-MEM/Reflexion behavior while exposing stateful, structured APIs for the newer architecture.

## Architecture

The affect subsystem sits on the input-processing path:

```text
Input Processing -> Affect Engine -> Persistent Cognitive State
```

It does not orchestrate retrieval, planning, reflection, memory consolidation, or memory writes. It emits structured metadata those subsystems may consume.

The memory subsystem sits between persistent cognitive state and expanded hybrid retrieval:

```text
Persistent Cognitive State -> Retrieval Controller -> Hierarchical Memory Fabric -> Expanded Hybrid Retrieval
```

It does not call planner, reflection, action, or LLM layers directly.

The reflection subsystem sits after retrieval confidence and before memory storage/future retrieval:

```text
Retrieval -> Retrieval Confidence -> Adaptive Reflection -> Reflection Memories -> Memory Fabric
```

It emits reflection signals, reflection memories, rules, confidence estimates, and state updates. It does not directly control retrieval, affect, memory consolidation, planning, or action execution.

## Package Layout

```text
affect/
  affect_engine.py              # DynamicAffectEngine entrypoint
  affect_state.py               # compatibility exports
  affect_types.py               # AffectUpdate and reflection signals
  emotion_classifier.py         # legacy-compatible classifier and cache hook
  emotion_profile.py            # immutable single-input profile
  emotion_history.py            # rolling history with velocity/acceleration/drift
  pad_model.py                  # immutable PADState model
  affect_evolution.py           # momentum, volatility, dissonance
  emotional_memory_adapter.py   # metadata generation only
  salience_modulation.py        # salience score
  retrieval_priors.py           # retrieval priors, no retrieval calls
  reflection_triggers.py        # reflection signals, no reflection calls
  emotion_cache.py              # lightweight LRU cache
  interfaces.py                 # Numpy/Faiss similarity backend abstraction
state/
  emotional_state.py            # persistent emotional state
  cognitive_state.py            # integration container
tests/
  affect/                       # deterministic unit tests
memory/
  memory_note.py                 # metadata-rich memory note and A-MEM formatter
  memory_repository.py           # Chroma/in-memory repository layer
  memory_store.py                # logical store wrapper
  stores/                        # working, episodic, semantic, emotional stores
  graph/                         # persistent graph nodes, edges, traversal, reasoning
  retrieval/                     # dense, sparse, temporal, graph, fusion, confidence
  evolution/                     # semantic abstraction, lineage, evolution engine
  maintenance/                   # salience, retention, soft forgetting, consolidation
reflection/
  adaptive_reflection_pipeline.py # verifier -> reflection -> retry -> ExpeL flow
  reflection_engine.py           # state-aware trigger scoring and signal/memory output
  reflection_context.py          # query, memory, affect, state, confidence context
  verifier_adapter.py            # PASS parsing, fuzzy matching, grounding checks
  failure_classifier.py          # typed failure source classification
  rule_extractor.py              # ExpeL-style rule distillation
  reflection_repository.py       # reflection/rule repository with Memory Fabric adapter
tests/
  retrieval/
  graph/
  evolution/
  lineage/
  reflection/
```

## Usage

```python
from affect import DynamicAffectEngine

engine = DynamicAffectEngine()
update = engine.process("I am really scared and nervous about my exam")

print(update.profile.dominant_emotion)
print(update.retrieval_priors)
print(update.reflection_signals)
print(update.memory_metadata)
```

Backward-compatible helper:

```python
from affect import get_emotion_profile

profile = get_emotion_profile("I feel very happy today")
```

Memory fabric:

```python
from memory import InMemoryMemoryRepository
from memory.memory_note import MemoryNote
from memory.retrieval.retrieval_controller import RetrievalController
from memory.retrieval.retrieval_request import RetrievalRequest

repository = InMemoryMemoryRepository()
repository.add(MemoryNote.create("I baked sourdough bread for the party"))

response = RetrievalController(repository).retrieve(RetrievalRequest(query="sourdough party"))
print(response.results[0].note.content)
print(response.confidence)
```

Adaptive reflection:

```python
from reflection import AdaptiveReflectionPipeline

pipeline = AdaptiveReflectionPipeline()
result = pipeline.run_answer_trial(
    query="Who was Milhouse named after?",
    proposals=["Abraham Lincoln", "Richard Nixon"],
    ground_truth="Richard Nixon",
    retrieved_content="Milhouse was named after Richard Nixon.",
)

print(result.is_correct)
print(result.extracted_rules)
```

## Design Notes

- Deterministic local fallback classifier is included so tests and PRIMA state behavior do not require network downloads.
- Legacy scoring semantics are preserved: nearest-neighbor rank weighting, score normalization, intensity modifiers, and Plutchik-style negation flipping.
- FAISS is optional. The default backend is NumPy.
- `EmotionalMemoryAdapter` only creates metadata. It never writes memory.
- Retrieval priors and reflection signals are emitted as data, not acted on directly.
- Memory fabric preserves dominant context chain extraction, smart keyword overlap boosting, hybrid retrieval, graph clustering, semantic evolution, and lineage.
- ChromaDB is the intended persistence source of truth; an in-memory repository is included for deterministic tests and local development.
- Forgetting is soft: maintenance lowers retention and suppresses retrieval rather than hard-deleting notes.
- Reflection preserves verifier-driven retry behavior, fuzzy answer matching, grounding validation, rejected-action feedback, and ExpeL rule extraction.
- Reflection consumes affect signals and retrieval confidence as data; it does not call affect or retrieval internals.

## Setup

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
python -m unittest discover
```

Optional legacy benchmark dependencies are listed as comments in `requirements.txt` because this repo should remain lightweight by default.

Runtime knobs live in `.env`; `.env.example` documents the expected keys.
