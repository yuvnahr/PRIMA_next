# PRIMA Demo CLI runbook

Run these commands from the repository root on branch `demo-cli`. PowerShell is the primary path; Git Bash alternatives are below. Configuration examples contain placeholders only. Enter credentials privately; never paste real keys into commands, screenshots, logs, or tracked files.

## 1. Prepare the environment

Open PowerShell:

```powershell
Set-Location C:\Users\prate\Code\GitHub\PRIMA_next
git branch --show-current
git status --short --ignore-submodules=all
git log -3 --oneline
python --version
```

Use Python 3.10 or newer. Prefer a patched Python release over 3.10.0. If you are on another branch, preserve/review local changes before switching to `demo-cli`; do not reset or clean the working tree to follow this guide.

Create a virtual environment if needed, then activate it:

```powershell
if (-not (Test-Path -LiteralPath .venv\Scripts\python.exe)) {
    python -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable); print(sys.version)"
```

If PowerShell blocks activation, allow local scripts for this terminal process only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
```

Install the demo dependencies **inside that environment**. The demo requirements combine the core runtime and the tested Langfuse/OTel cohort in one pip resolution, without the general requirements file's benchmark/development tools:

```powershell
python -m pip install -r requirements-demo.txt
python -m pip check
python -m pip show langfuse opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http opentelemetry-exporter-otlp-proto-grpc chromadb
```

The optional observability file pins Langfuse **4.18.0** and OTel API/SDK plus HTTP/gRPC exporters **1.45.2**. Keep this cohort together: Chroma imports the gRPC exporter too. Do not replace it with an unpinned SDK upgrade. The old Langfuse 2.x installation is not the supported tracing path. The ignored `.codex_deps/observability` audit cache is not loaded automatically by the demo.

Without Langfuse, only core requirements are needed. The launchers install nothing. The broader `setup.ps1` described in [README.md](README.md#setup) also installs development/benchmark dependencies and initializes submodules; that larger setup is not required just to launch the demo.

## 2. Configure Groq and Langfuse

Create `.env` only if it does not already exist, then open it privately:

```powershell
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
git check-ignore .env
notepad .env
```

Set these values in the editor. Replace placeholders with your own credentials, and use the Langfuse URL for your project's region:

```dotenv
DEFAULT_PROVIDER=groq
DEFAULT_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_URL=https://api.groq.com/openai

LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=YOUR_LANGFUSE_PROJECT_PUBLIC_KEY
LANGFUSE_SECRET_KEY=YOUR_LANGFUSE_PROJECT_SECRET_KEY
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=demo
```

Keep `LANGFUSE_ENABLED=false` to enable tracing explicitly with `--langfuse` in this guide. Set it to `true` if every invocation should enable tracing. Process environment values override `.env`; CLI provider/model flags override project settings. Restart the CLI after configuration changes.

In Langfuse, create/select your project and obtain its public/secret **project API key pair** from project settings. The keys must belong to the same project and region you will inspect. Use the base URL without `/api` or an ingestion path. [Official credential and region instructions](https://langfuse.com/docs/api-and-data-platform/features/public-api).

| Langfuse deployment | `LANGFUSE_BASE_URL` |
|---|---|
| Cloud EU | `https://cloud.langfuse.com` |
| Cloud US | `https://us.cloud.langfuse.com` |
| Cloud Japan | `https://jp.cloud.langfuse.com` |
| Self-hosted | Your instance's reachable root URL |

For the lightweight demo, the runtime defaults to the legacy affect classifier, stable local embeddings, and lexical reranking. If your terminal already overrides these settings, select the lightweight configuration explicitly for a **fresh** demo:

```powershell
$env:PRIMA_AFFECT_BACKEND = 'legacy'
$env:PRIMA_EMBEDDING_BACKEND = 'stable'
$env:PRIMA_RERANKER_BACKEND = 'lexical_fallback'
```

These runtime overrides are process environment settings. Do not change embedding configuration when reopening existing persistent data; stored fingerprints must match.

## 3. Inspect help and run local preflight

These commands do not make a model request:

```powershell
python -m demo_cli.cli --help
python -m demo_cli.cli chat --help
python -m demo_cli.cli preflight --help
.\demo.ps1 status --provider groq --model openai/gpt-oss-120b --no-color
.\demo.ps1 preflight --provider groq --model openai/gpt-oss-120b --langfuse --no-color
$LASTEXITCODE
```

Expect `groq`, `openai/gpt-oss-120b`, `prima_full`, an in-memory backend, and a configured Groq key. With valid local tracing configuration/dependencies, preflight reports Langfuse `ready`. It does **not** authenticate with Groq/Langfuse, generate a trace, or prove export delivery. A tracing warning is optional and permits chat.

| Result | Meaning |
|---|---|
| Exit `0`, READY | Required local checks passed |
| Exit `0`, READY WITH WARNING | Required checks passed; inspect optional tracing warning |
| Exit `1`, NOT READY | Fix the displayed configuration/dependency/storage issue |
| Exit `2` | Invalid command-line syntax; inspect `--help` |

`status` reports construction/configuration, not provider health. Its Langfuse status can be `disabled`, `enabled`, or `misconfigured`; `enabled` means SDK initialization succeeded. To inspect an invocation enabled with `--langfuse`, use `/status` **inside that chat**. Standalone `status` does not accept `--langfuse`.

### Optional Groq connectivity diagnostic — consumes a provider request

Skip strict preflight when the planned chat will verify connectivity anyway. Use it only when troubleshooting requires an isolated generation:

```powershell
.\demo.ps1 preflight --strict --provider groq --model openai/gpt-oss-120b --langfuse --no-color
```

This makes one generation with an eight-token limit, ten-second timeout, and zero client retries. It can consume hosted tokens. It does not exercise a full PRIMA turn or prove Langfuse export, and a reasoning model may use the tiny budget without producing useful prose.

## 4. Launch the demo

### Groq with Langfuse and live terminal events

```powershell
.\demo.ps1 chat --provider groq --model openai/gpt-oss-120b --profile prima --session capstone-demo-01 --trace summary --live-trace --langfuse --temperature 0 --max-tokens 256 --no-color
```

Use a fresh, non-sensitive ASCII session label shorter than 200 characters, and keep the exact label for finding traces. Reusing it groups later traces under the same Langfuse session; it does not itself enable local persistence.

The explicit 256-token cap keeps the smoke test small while allowing reasoning and a short answer. The application's default is 4096 if you omit the flag. Every conversational prompt can consume provider tokens; inspection commands below do not. Client retries default to zero, but invalid generated output can still trigger bounded output correction and another request. Do not repeat authentication/configuration failures unchanged.

### Groq without Langfuse

Ensure `LANGFUSE_ENABLED=false` in `.env` and any process override, then omit `--langfuse`:

```powershell
.\demo.ps1 chat --provider groq --model openai/gpt-oss-120b --profile prima --session capstone-offline-traces-01 --trace summary --live-trace --temperature 0 --max-tokens 256 --no-color
```

There is no interactive `/langfuse` or `--no-langfuse` switch. Disable the setting and restart. Live terminal events and post-turn diagnostics continue to work without Langfuse.

If script execution is blocked, an equivalent module invocation is:

```powershell
python -m demo_cli.cli chat --provider groq --model openai/gpt-oss-120b --profile prima --session capstone-demo-01 --trace summary --live-trace --langfuse --temperature 0 --max-tokens 256 --no-color
```

## 5. Run the two-turn presentation and inspect intermediate results

At `You >`, enter the following **text**, not a PowerShell command:

```text
I have my capstone demo today and I'm nervous because my last presentation went badly. Please remember this concern. Give me only two short things to focus on right before I start.
```

After the answer, enter these commands one at a time:

```text
/status
/state
/memory
/plan
/world
```

Check that the provider/model/session are correct, the state version advanced, and actual affect, plan, world, uncertainty and decision results appear. Memory admission is threshold-based: inspect the reported decision rather than assuming everything is stored. The explicit remember sentence above was used in the successful audit to establish continuity without changing admission policy.

Enter the second prompt:

```text
What did I say was making me nervous, and give me one thing to focus on now. Keep the answer to two short sentences.
```

Then inspect:

```text
/state
/memory
/plan
/world
```

Expect the same session, a second state-version advance, and earlier relevant memories in the retrieval summary where admitted. In the audited run, versions advanced `0 -> 1 -> 2`, two first-turn memories were retrieved on turn two, and the second turn itself was not admitted. Generated wording and admission are not guaranteed for arbitrary prompts.

### All interactive commands

| Command | Purpose / effect |
|---|---|
| `/help` | List commands |
| `/status` | Provider/model/profile/session/backend, persistence, live/diagnostic modes and Langfuse initialization status |
| `/state` | Actual current state/version and persisted affect/PAD; salience is not a stored state field |
| `/memory` | Up to five recent stored memories; lists notes without semantic retrieval |
| `/plan` | Latest turn's public plan, actions and constraints |
| `/world` | Latest symbolic forecasts, risk/failure probabilities, effects and warnings; not confirmed real-world violations |
| `/trace` | Show current post-turn diagnostic mode |
| `/trace off`, `/trace summary`, `/trace full` | Control diagnostics printed after subsequent turns; full includes bounded details/events |
| `/live`, `/live on`, `/live off` | Show/change foreground workflow event display for subsequent turns |
| `/profile` | Show current profile and choices |
| `/profile prima`, `/profile rag`, `/profile model` | Change future execution to PRIMA_FULL, SIMPLE_RAG or MODEL_ONLY; keep provider/model/session/state |
| `/clear` | Clear presentation output only |
| `/reset` | Fresh runtime, affect history and session ID; see persistent-mode deletion warning below |
| `/quit`, `/exit` | Drain/stop owned work, detach the observer, close stores and attempt bounded telemetry shutdown |

These commands do not invoke the LLM. Profile changes do not rerun the last prompt; a later conversation prompt is a new billable request. `/plan` and `/world` are unavailable before a first turn and remain response-local, including after a persistent reopen. Failed new turns do not leave old outputs masquerading as current.

### Understand the three tracing views

| View | Source | Enable / inspect |
|---|---|---|
| Live terminal events | Actual foreground EventBus workflow events | `--live-trace`, `/live on` |
| Post-turn diagnostics | Actual `PrimaResponse` diagnostics | `--trace summary` or `/trace full` |
| Langfuse traces | Optional OTel export of canonical runtime spans | Configure SDK/keys/URL, use `--langfuse`, inspect the project UI |

`[OK]` means a controller emitted phase completion, not guaranteed provider success. Final errors/outcomes remain authoritative. Live display preserves repeated phases, prints at most 50 selected events per turn, omits absent metrics/skips, and excludes background maintenance. Reflection is selective; `not executed` is valid. No tokens are streamed.

## 6. Find and check Langfuse traces

1. After at least one conversation turn, enter `/status` and confirm Langfuse is `enabled`. Merely starting chat or running inspection commands does not create a canonical turn trace.
2. Enter `/quit` once to finish cleanup and attempt export of queued data. Export is asynchronous; shutdown waits at most three seconds and can drop pending traces on timeout.
3. Open the same regional/self-hosted Langfuse UI used in `LANGFUSE_BASE_URL`; select the project that owns your API keys.
4. Open **Observability / Observations** (or **Tracing**, depending on your deployment). Select the recent time range and environment `demo`, matching `LANGFUSE_TRACING_ENVIRONMENT`.
5. Filter by observation name **`prima.turn`** and by **Session ID** matching the exact CLI label, such as `capstone-demo-01`. Refresh after allowing export/ingestion time. On current Langfuse versions the observations table can default to root observations only; open the root trace to inspect its children. [Official observations-table guide](https://langfuse.com/faq/all/explore-observations-in-v4).
6. Open a matching root and inspect its tree/timeline, attributes, outcome, duration and generation usage. For multiple turns, use the trace's session badge or the Sessions view to inspect the shared session. `/reset` moves subsequent turns to a new session ID. [Official sessions guide](https://langfuse.com/docs/observability/features/sessions).

Normal PRIMA_FULL traces contain the actual executed phases below; reflection/action are conditional, stages can repeat, and profile skips do not create placeholder spans:

```text
prima.turn
  state_load
  affect
  evidence_acquisition
    memory_retrieval       (one span per actual retrieval call)
  planning
  world_simulation
  uncertainty_estimation
  execution_decision
  reflection               (only if routed)
  action                   (only if routed)
  answer_generation
    llm.generation         (one span per actual provider attempt)
  output_validation
  output
  state_commit
  memory_commit
  maintenance_enqueue
```

Check these PRIMA attributes/metadata when present; UI placement depends on the Langfuse version:

| Span | Useful fields |
|---|---|
| `prima.turn` | `prima.session_id`, `prima.request_id`, `prima.execution_id`, `prima.execution_profile`, `prima.provider`, `prima.model`, `prima.status`, `prima.outcome`, `prima.state_version_before`, `prima.state_version_after`, `prima.latency_ms` |
| `affect` | Actual dominant emotion, PAD, confidence, transient salience, stability/volatility |
| `evidence_acquisition` / `memory_retrieval` | Candidate/selected counts, retrieval confidence, final context count on evidence acquisition |
| `planning` | Action/constraint counts, confidence, revision |
| `world_simulation` | Risk/failure probabilities and prediction/warning counts |
| `uncertainty_estimation` / `execution_decision` | Actual uncertainty score/band and decision |
| `llm.generation` | Model, parameters, actual reported token usage and attempt; also check failed status |
| `state_commit` / `memory_commit` | Whether state committed, resulting version, admission and created-note count |

The UI intentionally has **no prompt, answer or memory content** for these traces. PRIMA exports metadata only; an empty input/output pane is expected. Session grouping does not enable content replay. Use the terminal to inspect answer text and memory previews. Missing token usage must not be treated as zero; enqueue is not proof that cold maintenance finished.

Do not declare Langfuse delivery verified until you can see the new `prima.turn` and its actual children in the correct project. The prior audit validated the modern SDK locally; it did not validate cloud delivery.

## 7. Diagnostics when Langfuse traces are missing

First recheck the active Python environment and local SDK readiness without sending a Groq request:

```powershell
python -c "import sys; print(sys.executable)"
python -m pip check
python -m pip show langfuse opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http opentelemetry-exporter-otlp-proto-grpc
python -c "from langfuse import Langfuse, LangfuseGeneration, propagate_attributes; print('Required SDK imports: OK')"
python -c "from config.settings import get_settings; from monitoring.tracing import langfuse_readiness; print(langfuse_readiness(get_settings(), enabled=True))"
.\demo.ps1 preflight --provider groq --model openai/gpt-oss-120b --langfuse --no-color
```

If imports fail or versions conflict, activate the intended environment and reinstall `requirements-observability.txt`; check `pip check` again. Do not dump `.env`, all environment variables, HTTP headers or SDK debug logs onto a presentation screen.

### Optional Langfuse authentication check — network, no Groq generation

This explicitly enables the configured Langfuse client, uses its public authentication check, prints only the result/error type, and runs PRIMA's bounded shutdown. It contacts Langfuse but does not submit a PRIMA prompt or create a synthetic turn. A successful authentication check still does not prove the trace export path. [Official authentication troubleshooting](https://langfuse.com/docs/observability/sdk/troubleshooting-and-faq).

```powershell
@'
from config.settings import get_settings
from monitoring.tracing import configure_tracing, shutdown_tracing

try:
    handle = configure_tracing(get_settings(), enabled=True)
    if handle.status != "enabled" or handle.client is None:
        print("Langfuse unavailable:", handle.warning or handle.status)
    else:
        print("Langfuse authentication:", handle.client.auth_check())
except Exception as error:
    print("Langfuse authentication check failed:", type(error).__name__)
finally:
    shutdown_tracing()
'@ | python -
```

| Symptom | Check / action |
|---|---|
| `disabled` | Add `--langfuse` to chat, or set `LANGFUSE_ENABLED=true` and restart |
| `misconfigured`, `unavailable`, import errors | Same interpreter/venv, supported SDK cohort, nonempty project keys, valid base URL |
| Authentication check fails | Correct project key pair and region; reachable URL/TLS/proxy/firewall. Do not retry bad keys unchanged. |
| Authentication works but no trace | Send an actual turn; confirm correct project/session/time/environment filters; wait for asynchronous ingestion and inspect shutdown warnings |
| Root visible but no child rows in table | Open the trace tree or clear the root-observation-only table filter |
| Session not found | Exact ASCII session label; remove old UI filters; after reset use the new displayed ID |
| Empty input/output | Expected metadata-only policy, not missing instrumentation |
| Missing reflection span | Check actual routing/diagnostics; do not force extra paid prompts solely to trigger reflection |
| Generic export warning | PRIMA intentionally hides raw SDK/server diagnostics; continue chat if needed, investigate configuration privately |

There is no CLI command that fetches, replays or deletes Langfuse traces, and no trace-ID display command. Use the project UI and session filter.

## 8. Optional persistence and resume

Persistence is off by default, even when `--session` is supplied. To demonstrate it, use a dedicated session and **one process/writer per session**:

```powershell
.\demo.ps1 preflight --persist --session capstone-persist-01 --provider groq --model openai/gpt-oss-120b --langfuse --no-color
.\demo.ps1 chat --persist --session capstone-persist-01 --provider groq --model openai/gpt-oss-120b --live-trace --langfuse --temperature 0 --max-tokens 256 --no-color
```

Inspect `/state` and `/memory`, enter turns if needed, then `/quit`. Run the same chat command again to resume; inspection itself makes no model request. Default non-strict persistent preflight checks advisory permissions/existing JSON without opening Chroma or creating session data. `--strict` additionally opens stores and makes the small provider request described above.

The default root is ignored `.demo_data`. To choose another root, add **the same** `--data-dir 'C:\path\to\demo-data'` to preflight and every resume invocation, or configure `PRIMA_DEMO_DATA_DIR`. Relative roots resolve from the repository. Data is scoped to `sessions/<SHA-256-of-session-id>/state.json` and `chroma/`.

Persisted cognitive state, PAD/momentum, version and admitted memories survive reopening. Rolling affect history, last `/plan` and last `/world` are process-local. A missing explicit session ID creates a new ID and displays its resume command; it does not automatically pick old data.

**Persistent `/reset` deletes the active old session's state/memories/graph and starts a new session ID.** Other sessions remain untouched. JSON and Chroma are not transactional together: partial deletion can occur; on failure the session closes and does not claim success. Inspect storage before reusing that ID. Stores are local/unencrypted; keep custom roots outside Git. Fingerprint mismatches fail without automatic rebuilding or deletion.

## 9. Ollama fallback and Git Bash

To use Ollama, exit Groq chat first. In a terminal with Ollama installed, list the exact installed model name; if its server is not running, start it in a separate terminal:

```powershell
ollama list
# Only if the server is not already running:
ollama serve
```

In the demo terminal, replace `EXACT-INSTALLED-MODEL-ID` with a name from that list:

```powershell
.\demo.ps1 preflight --provider ollama --model 'EXACT-INSTALLED-MODEL-ID' --langfuse --no-color
.\demo.ps1 chat --provider ollama --model 'EXACT-INSTALLED-MODEL-ID' --session capstone-local-01 --live-trace --langfuse --max-tokens 256 --no-color
```

Non-strict preflight does not check server reachability. Optional `preflight --strict` generates locally to check the server/model. No Groq tokens are used on this provider. The demo does not download models or switch providers/models inside a session. Omit `--langfuse` with the setting disabled if tracing is unavailable.

Git Bash on Windows uses the same configuration and CLI flags:

```bash
cd /c/Users/prate/Code/GitHub/PRIMA_next
source .venv/Scripts/activate
./demo preflight --provider groq --model openai/gpt-oss-120b --langfuse --no-color
./demo chat --provider groq --model openai/gpt-oss-120b --profile prima --session capstone-bash-01 --trace summary --live-trace --langfuse --temperature 0 --max-tokens 256 --no-color
```

Use `bash ./demo ...` if executable permissions prevent `./demo`. On POSIX systems, use your checkout path and `.venv/bin/activate` instead.

## 10. Reference, regression diagnostics and exit

Common flags accepted by `chat`, `status` and `preflight`: `--provider`, `--model`, `--profile prima|rag|model`, `--session`, `--trace off|summary|full`, `--temperature`, `--max-tokens`, `--persist`, `--data-dir`, `--no-color`. `--live-trace` is chat-only; `--strict` is preflight-only; `--langfuse` is available on chat/preflight.

| Default / process override | Value |
|---|---|
| `PRIMA_DEMO_PROFILE` | `prima` |
| `PRIMA_DEMO_TRACE` | `summary` |
| `PRIMA_DEMO_SESSION` | Generated ID when unset |
| `PRIMA_DEMO_TEMPERATURE` | `0` |
| `PRIMA_DEMO_MAX_TOKENS` | `4096` |
| `PRIMA_DEMO_DATA_DIR` | `.demo_data`, also supported by project settings |
| `NO_COLOR` | Presence disables color |

The profile/trace/session/temperature/token demo overrides above come from the process environment; provider/model/Langfuse settings also load from `.env`. There is no live-trace environment setting. Flags take precedence where implemented.

For developer diagnostics, install the separate development tools if needed, then run offline tests using fake providers:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q tests/demo_cli tests/monitoring tests/runtime tests/test_groq_provider.py --tb=short
python -m pytest -q --ignore=tests/test_production_embedding_smoke.py --tb=short
git diff --check --ignore-submodules=all
```

The last broad suite excludes the real embedding smoke test. Audit results and exact environments are in the local ignored `handoff.md` when present: final CLI 171 passed; broad normal 567 passed, five classified benchmark failures and one old-SDK skip; modern broad 568 passed with the same five benchmark failures. These are historical results, not a promise for a newly installed environment. Windows sandbox restrictions can affect asyncio tests; use an appropriately authorized local terminal rather than diagnosing a sandbox hang as a runtime regression.

Finish chat with `/quit` or `/exit`. Windows EOF is Ctrl+Z then Enter; Git Bash EOF is Ctrl+D. Ctrl+C requests cleanup, but an active HTTP worker may continue until its timeout (normally 60 seconds). Avoid repeated interrupts. Synchronous idle input can delay maintenance; normal quit drains it. Telemetry shutdown is bounded to three seconds and may drop queued exports.

After leaving the demo, close any trace/credential editor before sharing the screen. If credentials were supplied through process environment variables, clear them when finished; this does not remove credentials you separately saved in `.env`:

```powershell
Remove-Item -LiteralPath Env:\GROQ_API_KEY, Env:\LANGFUSE_PUBLIC_KEY, Env:\LANGFUSE_SECRET_KEY -ErrorAction SilentlyContinue
deactivate
```

The last real Groq audit passed with two turns/two requests at a 256-token cap. Live Langfuse cloud delivery still needs confirmation in your configured project using section 6.
