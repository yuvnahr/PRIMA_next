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

The planning layer sits between memory retrieval and reflection/action:

```text
Persistent Cognitive State + Retrieved Memory Context + Affective Priors + Reflection Signals -> Task Planner -> Plan
```

It emits structured plans, actions, execution intent, constraints, simulations, and replanning lineage. It does not execute tools, invoke LLMs, or orchestrate other subsystems.

The action layer sits after planning and reflection:

```text
Plan -> Action Executor -> Policy-Gated Tool Router -> Registered Tool Handler -> Execution Result
```

It converts plans into bounded execution results. It validates arguments, enforces sandbox policy, audits tool calls, and only dispatches explicitly registered handlers. It does not allow arbitrary command execution or bypass validation.

The event layer provides in-process asynchronous communication:

```text
Subsystem Output -> Event Publisher -> Async Event Bus -> Filtered Subscribers -> Event Store
```

It replaces direct module coupling with typed events such as memory creation, reflection triggers, state changes, plan failures, and tool execution. It uses asyncio and an in-memory store; no external broker is required.

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
planning/
  task_planner.py                 # pure planner facade with replanning support
  plan.py                         # immutable plan, action, constraint, intent, simulation models
  planning_context.py             # workflow-routed context adapter
  goal_selector.py                # state-aware goal selection
  action_selector.py              # pure action and execution intent selection
  plan_evaluator.py               # deterministic simulation and plan evaluation
  planning_types.py               # planning enums
tests/
  retrieval/
  graph/
  evolution/
  lineage/
  reflection/
workflow/
  prima_workflow.py               # high-level Cognitive Control Bus facade
  orchestration_engine.py         # lifecycle, retries, cancellation, event publication
  execution_context.py            # per-request state and routed subsystem outputs
  workflow_state.py               # phase/status models
  task_router.py                  # phase routing
  controller_registry.py          # dependency-injected controller pattern
  workflow_events.py              # async in-process event bus
action/
  action_executor.py              # plan-to-execution facade
  action_context.py               # workflow-provided action context and audit records
  execution_result.py             # aggregate and per-step execution results
  tool_invocation.py              # typed tool/external/environment invocation requests
  execution_policy.py             # sandbox policy and policy decisions
tools/
  tool_router.py                  # explicit registry-based routing
  tool_registry.py                # registered safe tool handlers and schemas
  tool_executor.py                # policy, validation, timeout, audit, execution
  tool_validator.py               # declarative argument validation and sanitization
  tool_result.py                  # structured tool execution results
events/
  event.py                        # immutable event envelope
  event_bus.py                    # asyncio in-process publish/subscribe bus
  event_types.py                  # canonical event type enum
  subscribers.py                  # filtered subscriber records and registry
  publishers.py                   # helper publisher factories for common events
  event_store.py                  # append-only in-memory event history
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

Pure planning:

```python
from planning import PlanningContext, TaskPlanner

context = PlanningContext(objective="Answer using retrieved memory")
plan = TaskPlanner().create_plan(context)

print(plan.execution_intent.intent_type)
print(plan.simulation.predicted_confidence if plan.simulation else None)
```

Cognitive workflow:

```python
from affect import DynamicAffectEngine
from memory import InMemoryMemoryRepository
from memory.retrieval.retrieval_controller import RetrievalController
from reflection import ReflectionEngine
from workflow import PrimaWorkflow

repository = InMemoryMemoryRepository()
workflow = PrimaWorkflow.from_controllers(
    affect_engine=DynamicAffectEngine(),
    retrieval_controller=RetrievalController(repository),
    reflection_engine=ReflectionEngine(),
)

context = await workflow.run("I am nervous about tomorrow")
print(context.output)
```

Policy-gated action execution:

```python
from action import ActionContext, ActionExecutor, ExecutionPolicy

policy = ExecutionPolicy(
    allowed_tools=("safe_lookup",),
    allow_external_actions=True,
    allowed_sandbox_tags=("read_only",),
)
result = await ActionExecutor(tool_executor=my_tool_executor).execute(ActionContext(plan=plan, policy=policy))

print(result.status)
print(result.audit_log)
```

Async events:

```python
from events import EventBus, EventPublisher, EventType

bus = EventBus()
bus.subscribe(handle_memory_created, event_types=(EventType.MEMORY_CREATED,), topics=("memory",))

publisher = EventPublisher(source="memory.repository", topic="memory")
await bus.publish(publisher.memory_created("mem_123"))
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
- Planning consumes workflow-routed state, retrieved memory summaries, affective priors, and reflection signals as data.
- Planning produces structured plans with constraints, simulated transitions, evaluation scores, and replanning lineage.
- Planning is pure reasoning: it never executes tools, dispatches actions, invokes LLMs, or writes memory.
- Workflow is the sole subsystem coordinator: user input flows through affect, memory retrieval, planning, reflection, action, and output via injected controllers.
- Workflow owns lifecycle state, execution context, event publication, retries, and cooperative cancellation.
- Action execution is sandbox-first: external tools are blocked by default, registered handlers are allowlisted by policy, arguments are schema-validated, and every invocation is audited.
- Tool execution supports tool calls, external actions, and environment operations only through typed `ToolInvocationKind` values and registered handlers; there is no arbitrary execution path.
- Events are local and async: the shared bus uses `asyncio`, filtered subscribers, typed event envelopes, and an injected in-memory event store. Kafka, Redis, RabbitMQ, and other external brokers are intentionally out of scope for now.

## Setup

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
python -m unittest discover
```

Rule-set checks after installing development dependencies:

```powershell
python -m pytest -q
python -m ruff check .
python -m mypy .
python -m bandit -r .
```

Optional legacy benchmark dependencies are listed as comments in `requirements.txt` because this repo should remain lightweight by default.

Runtime knobs live in `.env`; `.env.example` documents the expected keys.
