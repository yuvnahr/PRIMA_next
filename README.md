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

### Install core requirements

Activate your Python 3.10+ environment (the setup script uses `.venv`) and install:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
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

The CLI keeps one canonical `PrimaRuntime` and one session ID until `/reset`. Memory and cognitive state are in memory and last only for the running process; `--session capstone-demo` labels the session without enabling persistence. Provider/model selection stays fixed until exit.

### Architecture commands

| Command | Behavior |
|---|---|
| `/help`, `/status` | Show commands or active configuration. |
| `/trace off\|summary\|full` | Change display only; diagnostic collection remains enabled. |
| `/state` | Read the active session's current state manager, including evolved PAD, affect confidence, stability and volatility. Empty goal/task/confidence sections show `n/a`; these values are not reconstructed from a previous answer. |
| `/memory` | List up to five recent stored notes, with type, timestamp, salience and 240-character previews. This does not run retrieval. The repository is dedicated to the demo session; explicitly foreign-tagged notes are excluded, and untagged derived notes remain visible. |
| `/plan` | Show the latest public plan summary, with up to ten actions and ten constraints. |
| `/world` | Show the latest reported risk/failure probabilities, constraint-prediction count, up to ten predicted state deltas and ten warnings. Constraint predictions are not counted as confirmed violations. |
| `/profile prima\|rag\|model` | Change only future request profiles; no runtime/provider reconstruction or prompt rerun. `/profile` alone lists the current profile and choices. |
| `/clear` | Clear an ANSI-capable terminal, or scroll the viewport with blank lines; it never clears conversation data. |
| `/reset` | Drain/stop the old maintenance worker and replace only this demo's runtime, state and memory with a fresh session ID. Reuses the same LLM client/generation configuration and preserves the active profile and trace mode. |
| `/quit`, `/exit` | Drain maintenance and exit. |

`/plan` and `/world` identify the last turn's profile, report missing/skipped/failed components, and handle the period before a first turn. Switching profile preserves that last response until another request begins; reset discards it. An attempt that raises without returning a response leaves no retained plan/world output, so older results cannot appear as the latest turn. Inspection commands never call the LLM or rerun cognitive stages. Long values are bounded and omitted list entries are counted. State inspection uses PAD `pleasure` for the persisted valence dimension; salience is not stored in the current emotional-state model and is omitted. Predicted world effects are forecasts, not committed state changes.

All three subcommands accept `--provider`, `--model`, `--profile prima|rag|model`, `--session`, `--trace off|summary|full`, `--temperature`, `--max-tokens`, and `--no-color`. Flags override configuration. Provider/model settings reuse `DEFAULT_PROVIDER` and explicitly configured `DEFAULT_MODEL`; Groq otherwise defaults to `openai/gpt-oss-120b`, and Ollama requires an explicit model. Demo defaults are `prima`, `summary`, temperature `0`, and an output-token limit of `4096`. The matching `PRIMA_DEMO_PROFILE`, `PRIMA_DEMO_TRACE`, `PRIMA_DEMO_SESSION`, `PRIMA_DEMO_TEMPERATURE`, and `PRIMA_DEMO_MAX_TOKENS` environment variables can supply defaults. `NO_COLOR` disables color too. Provider/model settings load from `.env` as well as the environment; the `PRIMA_DEMO_*` overrides above come from the process environment.

Chat preflight validates provider/model configuration, credentials where required, and runtime construction; it does not probe provider availability or make a model call. `status` displays construction/configuration only, not provider health. Every turn collects diagnostic-mode data, even with trace display off. Full trace selects bounded public fields and at most 30 event summaries, excluding raw prompts, memory bodies, headers and event payloads.

Summary trace shows the actual profile, selected/candidate/context counts, plan confidence, world probabilities, uncertainty band/score, decision, reflection, memory admission and state-version transition where reported. Model-only turns use a shorter generation/commit/state summary. Profile skips, enabled-but-unexecuted stages, failures and unavailable diagnostics remain distinct; missing fields are not shown as zero measurements.

### Exit

Type `/quit` (or `/exit`). EOF also exits: Ctrl+Z then Enter in Windows consoles, Ctrl+D in Git Bash. Ctrl+C at the prompt exits gracefully. These paths drain/stop maintenance and attempt telemetry shutdown once. Console input is synchronous inside the single async session loop, so cold maintenance may wait while the prompt is idle. During a request, Ctrl+C requests shutdown; an already-running HTTP worker may still run until its configured timeout (normally 60 seconds). The CLI reports this limitation rather than claiming the HTTP request was cancelled. Telemetry shutdown waits at most three seconds; failure/timeout warns and preserves normal exit, but queued traces may be lost. Avoid repeated Ctrl+C during cleanup. Persistence and live workflow rendering remain separate future work.

See the short [presentation runbook](docs/demo_cli_runbook.md) for the operator checklist and suggested prompts.

## Optional Langfuse observability

Tracing is disabled by default. Install the optional, tested Langfuse/OpenTelemetry cohort into your project environment:

```powershell
python -m pip install -r requirements-observability.txt
```

This pins Langfuse 4.18.0 and OpenTelemetry API/SDK/HTTP/gRPC exporters 1.45.2, compatible with Python 3.10. The gRPC exporter is aligned because Chroma imports it; mixing its older release with SDK 1.45 breaks Chroma imports. Core requirements remain unchanged. An older Langfuse SDK is treated as unavailable rather than using legacy tracing APIs. Keep optional OpenTelemetry instrumentors/exporters aligned with this cohort when installing them separately.

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

Exported data is metadata only: identifiers, configuration labels, bounded aggregate results, reported token usage, timings and safe error types. No user text, generated answers, prompts, memory bodies, embeddings, arbitrary metadata, headers, exception messages or stack traces are attached. No content-capture switch is implemented. Use a non-sensitive session label; SDK session propagation requires ASCII IDs of at most 200 characters. Unsupported labels retain their bounded PRIMA metadata but omit Langfuse session grouping with a warning. Known configured keys are redacted from PRIMA string attributes. No token counts are estimated, and missing provider usage is omitted.

Applications embedding the runtime can call `monitoring.configure_tracing()` and `monitoring.shutdown_tracing()` at process lifecycle boundaries. Resetting a runtime must not shut down the shared telemetry setup. Tests inject isolated providers and in-memory exporters; no Langfuse service is needed.

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
