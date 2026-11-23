# PRIMA-NEXT

PRIMA-NEXT is an experimental Python runtime for building AI agents that can remember, reason, plan, reflect, and act through controlled tools.

In simple terms: a request enters one runtime, the runtime chooses the needed cognitive components, and it returns a typed answer with diagnostics. The project is designed for local research, reproducible benchmarks, and safe extension.

## What it includes

- Long- and short-term memory with dense, sparse, temporal, and graph retrieval.
- Emotion-aware state and retrieval signals.
- Planning, uncertainty checks, world simulation, and bounded reflection.
- Provider-backed language-model calls with structured output.
- Policy-gated tools with validation, timeouts, and audit records.
- HotpotQA, LoCoMo, and GoEmotions benchmark integrations.

PRIMA-NEXT is research software. It is not yet a hosted assistant or a production service.

## Setup

Requirements:

- Windows PowerShell
- CPython 3.14
- Git

From the repository root, run one command:

```powershell
.\setup.ps1
```

The script creates `.venv`, installs the full dependency set from `requirements.txt`, initializes Git submodules, and creates a local `.env` from `.env.example` when needed. It never overwrites an existing `.env`.

The Kaggle notebook uses Kaggle's managed Python kernel; that environment is separate from the local Python 3.14 environment and reports its own version in run artifacts.

If PowerShell blocks local scripts, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

## Quick start

Activate the environment and run the small affect demo:

```powershell
. .\.venv\Scripts\Activate.ps1
python main.py
```

Use the canonical runtime from Python:

```python
import asyncio

from runtime import ExecutionOptions, ExecutionProfile, PrimaRequest, PrimaRuntime, TaskKind


async def main() -> None:
    response = await PrimaRuntime().execute(
        PrimaRequest(
            task_kind=TaskKind.FACTUAL_QA,
            profile=ExecutionProfile.SIMPLE_RAG,
            input_text="What should I remember?",
            options=ExecutionOptions(max_retrieval_calls=1),
        )
    )
    print(response.output_text)


asyncio.run(main())
```

Configuration values and provider examples are documented in `.env.example`. The runtime has deterministic local fallbacks, so the test suite does not require a model download or API key.

## Demo CLI

### Install project requirements

Activate your Python 3.14 environment (the setup script uses `.venv`) and install:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Optional Langfuse

```powershell
python -m pip install -r requirements-observability.txt
```

Tracing is optional. Configure `LANGFUSE_ENABLED`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` in the environment or ignored `.env`; `--langfuse` enables one invocation. Missing packages/configuration or failed exports warn and let chat continue.

### Configure Groq

Set `DEFAULT_PROVIDER`, `DEFAULT_MODEL` and `GROQ_API_KEY` in your environment or ignored `.env`. See `.env.example` for provider/model examples. Flags override settings; no credentials belong in commands or source control.

### Preflight

```powershell
.\demo.ps1 preflight
# Optional: one small live provider request (uses tokens for hosted providers).
.\demo.ps1 preflight --strict
```

Default preflight checks imports, resolved configuration, required keys, real runtime construction and optional observability readiness. It makes no network requests. Strict adds one generation with at most eight output tokens, ten-second timeout and no retries. Ollama's default check is configuration only; strict checks reachability/model availability through generation. Langfuse readiness checks configuration/imports, not remote authentication or delivery, even in strict mode. Exit codes: `0` ready (including optional warnings), `1` not ready; argument syntax errors use argparse's `2`.

### Start

```powershell
.\demo.ps1 chat
.\demo.ps1 status
```

Launchers use `python` from the active environment, run from the repository root and pass all arguments to `python -m demo_cli.cli`. They install nothing and preserve Python's exit code. If local script policy blocks execution, use `powershell -ExecutionPolicy Bypass -File .\demo.ps1 preflight`. In Git Bash/POSIX shells, use `./demo preflight` and `./demo chat` (or `bash ./demo ...` if executable permissions are unavailable).

### Ollama fallback

Start your local Ollama server and supply an exact model ID already installed; the CLI never downloads models:

```powershell
.\demo.ps1 chat --provider ollama --model <exact-model-id>
```

Replace the angle-bracket placeholder with the actual ID before running.

The CLI keeps one canonical `PrimaRuntime` and one session ID until `/reset`. Provider/model selection stays fixed until exit.

### In-memory default

`.\demo.ps1 chat` starts fresh, process-local state and memory. Closing the process ends that state. `--session capstone-demo` alone labels the conversation without enabling persistence or initializing Chroma.

### Persistent session and resume

```powershell
.\demo.ps1 preflight --persist --session capstone-demo
.\demo.ps1 chat --persist --session capstone-demo
```

Quit and run the same chat command again to resume canonical cognitive state, affect/PAD and momentum, state version, and admitted memories. Without a supplied session ID, a new readable ID is generated and a resume command is displayed; no previous session is automatically selected. `/state` and `/memory` work immediately after reopening. Latest `/plan` and `/world` responses remain local to the current process. The affect engine's rolling history window also starts fresh; evolution still reads the persisted emotional state.

Data defaults to `.demo_data/sessions/<SHA-256-of-session-id>/state.json` and `chroma/`. Set `PRIMA_DEMO_DATA_DIR` in the environment/ignored `.env`, or override it with `--data-dir PATH`. Relative roots resolve from the repository. `.demo_data/` is ignored by Git; keep custom roots outside version control too. These stores contain real conversation memories and state. Use one writer per persistent session.

Persistent preflight checks local dependencies, existing state JSON and advisory directory permissions without creating session data or opening Chroma. `--strict` also opens/closes the stores and runs the existing tiny provider check; it never inserts test memories or advances state. Strict opening is the authoritative check for vector compatibility and actual backend access. No packages are installed automatically.

### Reset

`/reset` drains maintenance, clears only the active persistent session's canonical memories/graph and JSON state, closes its Chroma client, and starts a fresh persistent session ID. Provider/model/profile/trace/live-trace/Langfuse and data root stay unchanged. Reopening the old ID starts empty; other sessions are untouched. Empty backend files/directories may remain. JSON and Chroma do not share a transaction: a partial reset error closes the session and exits without claiming success; inspect storage before reusing that ID.

Stored embedding fingerprints must match the current embedding configuration; generation provider/model can change across launches independently. Incompatibility fails without rebuilding/deleting stored data. Reopen using the original embedding configuration to reset, or choose a new session ID. Chroma writes are durable as they occur; cleanup calls its public client `close()` where available. Older Chroma versions without that method manage resources until process exit.

### Live workflow trace

```powershell
.\demo.ps1 chat --live-trace
```

Live tracing is off by default. `/live`, `/live on`, and `/live off` show or change it for subsequent turns; `/status` reports the setting. It shows actual foreground workflow events while a turn executes. `/trace off|summary|full` independently controls completed post-turn diagnostics. Live output keeps emitted order and repeats, shows generation starting, and displays up to 50 events per turn. It omits values absent from events, skipped phases without skip events, raw payloads and background maintenance. `[OK]` means a phase-completion event; final request errors/diagnostics determine the turn outcome. No tokens are streamed. Live tracing works with either storage mode and independently of Langfuse.

### Architecture commands

| Command | Behavior |
|---|---|
| `/help`, `/status` | Show commands or active configuration. |
| `/trace off\|summary\|full` | Change display only; diagnostic collection remains enabled. |
| `/live [on\|off]` | Show or change live workflow event rendering for subsequent turns. |
| `/state` | Read the active session's current state manager, including evolved PAD, affect confidence, stability and volatility. Empty goal/task/confidence sections show `n/a`; these values are not reconstructed from a previous answer. |
| `/affect [detail]` | Inspect all emitted emotion scores and the last turn's affect update. Detail adds classifier cues, salience, momentum, retrieval priors, and reflection cues. |
| `/decision [detail]` | Read the actual gate path and rules alongside the estimator's four candidate probabilities. Detail adds signal weights, the confidence interval, and policy thresholds. |
| `/evidence` | Compare the latest retrieval selection with the final assembled context, with bounded excerpts. |
| `/memory` | List up to five recent stored notes, with type, timestamp, salience and 240-character previews. This does not run retrieval. The repository is dedicated to the demo session; explicitly foreign-tagged notes are excluded, and untagged derived notes remain visible. |
| `/memory why` | Inspect the latest admission decision, score versus threshold, and created note IDs. |
| `/action` | Inspect recorded action steps and tool or external submissions. |
| `/review` | Inspect reflection, accepted or rejected corrections, and technical validation. |
| `/plan` | Show the latest public plan summary, with up to ten actions and ten constraints. |
| `/world` | Show the latest reported risk/failure probabilities, constraint-prediction count, up to ten predicted state deltas and ten warnings. Constraint predictions are not counted as confirmed violations. |
| `/profile prima\|rag\|model` | Change only future request profiles; no runtime/provider reconstruction or prompt rerun. `/profile` alone lists the current profile and choices. |
| `/clear` | Clear an ANSI-capable terminal, or scroll the viewport with blank lines; it never clears conversation data. |
| `/reset` | Drain/stop maintenance and start a fresh session ID. In persistent mode, clear only the old active session's state/memory/graph. Preserve the provider/model, profile, trace/live settings, Langfuse, persistence mode and data root. |
| `/quit`, `/exit` | Drain maintenance and exit. |

`/plan` and `/world` identify the last turn's profile, report missing/skipped/failed components, and handle the period before a first turn. Switching profile preserves that last response until another request begins; reset discards it. An attempt that raises without returning a response leaves no retained plan/world output, so older results cannot appear as the latest turn. Inspection commands never call the LLM or rerun cognitive stages. Long values are bounded and omitted list entries are counted. State inspection uses PAD `pleasure` for the persisted valence dimension; salience is not stored in the current emotional-state model and is omitted. Predicted world effects are forecasts, not committed state changes.

All three subcommands accept `--provider`, `--model`, `--profile prima|rag|model`, `--session`, `--trace off|summary|full`, `--temperature`, `--max-tokens`, and `--no-color`. Flags override configuration. Provider/model settings reuse `DEFAULT_PROVIDER` and explicitly configured `DEFAULT_MODEL`; Groq otherwise defaults to `openai/gpt-oss-120b`, and Ollama requires an explicit model. Demo defaults are `prima`, `summary`, temperature `0`, and an output-token limit of `4096`. The matching `PRIMA_DEMO_PROFILE`, `PRIMA_DEMO_TRACE`, `PRIMA_DEMO_SESSION`, `PRIMA_DEMO_TEMPERATURE`, and `PRIMA_DEMO_MAX_TOKENS` environment variables can supply defaults. `NO_COLOR` disables color too. Provider/model settings load from `.env` as well as the environment; the `PRIMA_DEMO_*` overrides above come from the process environment.

Chat preflight validates provider/model configuration, credentials where required, and runtime construction; it does not probe provider availability or make a model call. `status` displays construction/configuration only, not provider health. Every turn collects diagnostic-mode data, even with trace display off. Full trace selects bounded public fields and at most 30 event summaries, excluding raw prompts, memory bodies, headers and event payloads.

Summary trace shows the actual profile, selected/candidate/context counts, plan confidence, world probabilities, uncertainty band/score, decision, reflection, memory admission and state-version transition where reported. Model-only turns use a shorter generation/commit/state summary. Profile skips, enabled-but-unexecuted stages, failures and unavailable diagnostics remain distinct; missing fields are not shown as zero measurements.

### Exit

Type `/quit` (or `/exit`). For EOF, use Ctrl+Z then Enter in Windows consoles or Ctrl+D in Git Bash. Ctrl+C at the prompt exits gracefully. These paths detach the live observer, drain or stop maintenance, close owned persistent resources, and attempt telemetry shutdown once. Console input is synchronous inside the single async session loop, so cold maintenance may wait while the prompt is idle. During a request, Ctrl+C requests shutdown. An already-running HTTP worker may still run until its timeout. The PRIMA workflow caps that timeout to the remaining turn budget. Telemetry shutdown waits at most three seconds, and timeout or failure can leave queued traces unsent. Avoid repeated Ctrl+C during cleanup.

See the short [presentation runbook](docs/demo_cli_runbook.md) for the operator checklist and suggested prompts.

## Optional Langfuse observability

Tracing is disabled by default. Install the optional, tested Langfuse/OpenTelemetry cohort into your project environment:

```powershell
python -m pip install -r requirements-observability.txt
```

This pins Langfuse 4.18.0 and OpenTelemetry API/SDK/HTTP/gRPC exporters 1.45.2. The gRPC exporter is aligned because Chroma imports it; mixing its older release with SDK 1.45 breaks Chroma imports. An older Langfuse SDK is treated as unavailable rather than using legacy tracing APIs. Keep optional OpenTelemetry instrumentors/exporters aligned with this cohort when installing them separately.

Configure credentials in your local environment or ignored `.env`:

```dotenv
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=
# LANGFUSE_TRACING_ENVIRONMENT=demo
```

Set `LANGFUSE_ENABLED=true` or enable just one chat invocation:

```powershell
python -m demo_cli.cli chat --provider groq --langfuse --session capstone-demo
```

`/status` reports `disabled`, `enabled` (initialized, not an authentication/availability check), or `misconfigured`. Missing configuration, unavailable SDKs and failed export never prevent a conversation. The existing CLI summary/full traces and inspection commands read runtime responses/state independently of Langfuse. Export is asynchronous; a failed export is diagnosed without dumping server responses or headers. `/quit`, EOF and Ctrl+C request SDK shutdown/flush once, with a three-second wait limit. On timeout, queued traces may be lost.

Each request creates one independent `prima.turn` root. For a normal full route, the actual hierarchy is:

```text
prima.turn
  state_load
  affect
  evidence_acquisition
    memory_retrieval (one per actual retrieval call)
  planning
  world_simulation
  uncertainty_estimation
  execution_decision
  reflection (only if routed)
  action (only if routed)
  answer_generation
    llm.generation (one per provider attempt)
  output_validation
  output
  state_commit
  memory_commit
  maintenance_enqueue
```

Retries/corrections can repeat stages. Model-only and RAG profiles produce only their real routed stages. `maintenance_enqueue` measures admission to the background queue, not completion of cold maintenance. Constraint-prediction counts are not confirmed violations. Affect salience is taken from the transient `AffectUpdate`, when available.

PRIMA reuses the existing OpenTelemetry provider or installs one when tracing is first enabled. Existing processors remain attached; Langfuse initialization adds one processor once per provider. Explicit `init_tracing()` still opts into console diagnostics and is idempotent. Langfuse exports only marked PRIMA spans from the `prima.runtime` scope, including generation observations wrapping those same spans; unrelated instrumentation is filtered out. Session grouping uses the SDK's official `propagate_attributes(session_id=...)`. `/reset` reuses this process-scoped tracing setup and propagates the fresh session ID. `/profile` affects future turn metadata and actual stage selection.

The default export is metadata only: identifiers, configuration labels, bounded aggregate results, reported token usage, timings and safe error types. Add `--trace-content` with `--langfuse` to export actual prompts, final assistant responses, selected memory content and structured phase inputs/outputs. `/status` shows the capture mode. Content capture is independent of terminal `--trace` and `--live-trace`.

```powershell
.\demo.ps1 chat --provider groq --model openai/gpt-oss-120b --langfuse --trace-content --max-tokens 4096
```

Content mode exports real text to your configured Langfuse project. Configured credentials and recognized credential patterns are redacted; headers, embeddings, arbitrary runtime objects and private model reasoning are excluded. Each input/output is bounded to a 131,072-character traversal budget, with individual strings limited to 65,536 characters, collections to 64 entries and nesting to 10 levels. Limits are explicitly marked; this is separate from the LLM completion-token limit. Use synthetic presentation data. The root shows the user request and final runtime answer; `llm.generation` shows actual logical request messages and provider assistant text before runtime shaping. Planning, world and action nodes show their real structured results. Symbolic predictions are not executed effects; technical validation is not a semantic answer-quality score. LLM-only model/token fields remain inapplicable to deterministic nodes.

Use a non-sensitive session label; SDK session propagation requires ASCII IDs of at most 200 characters. Unsupported labels retain bounded PRIMA metadata but omit Langfuse session grouping with a warning. No token counts are estimated, and missing provider usage is omitted. Applications embedding the runtime opt into content with `PrimaRuntime(trace_content=True)`; the default remains off even when telemetry is enabled.

### Qualitative base-model comparison

```powershell
.\demo.ps1 compare --provider groq --model openai/gpt-oss-120b --session capstone-compare --langfuse --trace-content --trace off --temperature 0 --max-tokens 4096 --no-color
```

Each entered message runs sequentially through ordinary model chat and a separate full PRIMA session. Both use the same model, generation settings, task-level policy, provider client and rate limiter. The base branch receives its own full user/assistant history; PRIMA uses its actual state and selected memories. The base branch has no cognitive workflow. OpenAI-compatible providers with role-based history (including Groq) are supported; unsupported adapters fail explicitly. No second key or dependency is needed.

Answers appear side by side at widths of at least 100 columns and stacked otherwise, with actual latency, provider-call counts, reported tokens and trace IDs. `/reset` clears both branches and changes the comparison group. All workflow inspection commands describe PRIMA in comparison mode. `/profile` is fixed. Compare mode is process-local and rejects `--persist`. Baseline input/history plus policy is limited to 100,000 characters; exceeding it stops both branches before a call and asks for reset, without silently dropping messages. This character bound does not guarantee an arbitrary provider model's context-window limit. A branch failure retains the other answer and requires a paired reset before more turns.

Langfuse receives two independent traces per paired turn: `base.turn` with a generation child, and `prima.turn` with its actual phases. Shared `comparison_id`, `comparison_turn` and `comparison_branch` metadata links the records; branch session IDs remain separate. Langfuse works without a graph framework. This is a qualitative demonstration of an application layer, not a claim that PRIMA always improves answers; the same model may answer equally well from ordinary history.

Applications embedding the runtime can call `monitoring.configure_tracing()` and `monitoring.shutdown_tracing()` at process lifecycle boundaries. Resetting a runtime must not shut down the shared telemetry setup. Tests inject isolated providers and in-memory exporters; no Langfuse service is needed.

In `compare`, type `/eval` after a paired response for a quick rubric estimate. A single judge request grades both answers on task fulfillment, correctness and grounding, context use and adaptation, and clarity and usefulness. It sees the full conversation so far with each branch's own replies. The CLI computes the score out of 100 and flags near ties. Groq defaults to `openai/gpt-oss-20b` as judge. Use `--judge-model MODEL` to choose a schema-capable model on the same provider. Repeated evaluations of a successfully graded pair reuse its score. The judge runs from local conversation data and works with Langfuse disabled.

## Development

Run the same quality checks used by CI:

```powershell
python -m benchmarks.preflight --json
python -m compileall -q -x '(^|[\\/])(\.git|\.venv|venv|external|pytest-tmp|tmp)([\\/]|$)' .
python -m pytest -q
python -m ruff check .
python scripts/run_xenon.py
python -m mypy .
python -m bandit -q -c bandit.yaml -r .
```

All runtime, benchmark, development, encoder/training, and semantic-metric dependencies are listed in `requirements.txt`. This includes PyTorch and BERTScore, so a fresh environment install is larger than the previous default. Benchmark instructions are in [benchmarks/README.md](benchmarks/README.md), and operational details are in [docs/benchmarks/RUNBOOK.md](docs/benchmarks/RUNBOOK.md).

## Project structure

- `runtime/` provides the public request/response boundary.
- `demo_cli/` provides the interactive chat REPL over that boundary.
- `workflow/` coordinates the selected cognitive components.
- `memory/`, `affect/`, `reasoning/`, `planning/`, and `reflection/` contain the core subsystems.
- `action/` and `tools/` enforce controlled execution.
- `benchmarks/` and `evaluation/` contain research and validation code.
- `tests/` contains deterministic unit, integration, architecture, and release checks.

## Contact and license

[Yuv Nahar](https://github.com/yuvnahr) is the project lead and primary contact. Everyone represented in the Git history is listed in [CONTRIBUTORS.md](CONTRIBUTORS.md).

PRIMA-NEXT is licensed under the [Apache License 2.0](LICENSE). Third-party dependencies, datasets, and submodules remain subject to their own licenses.
