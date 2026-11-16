# Demo CLI runbook

Use the normal in-memory mode for the main live presentation unless cross-process persistence is part of the demonstration.

## Before the demo

1. Open the repository in PowerShell and activate the intended environment: `. .\.venv\Scripts\Activate.ps1`.
2. Set provider/model and credentials in the environment or ignored `.env`; keep keys off screen.
3. Run `.\demo.ps1 preflight`. Fix required configuration errors. Optional Langfuse warnings permit chat.
4. Run `.\demo.ps1 preflight --strict` to verify the selected provider with one tiny generation. This can use hosted-provider tokens. For Ollama, use an exact installed model ID; do not download models on stage.
5. If using Langfuse, install `requirements-observability.txt`, configure its three credentials/URL variables, and enable `LANGFUSE_ENABLED`. Preflight checks local configuration/dependencies only; verify delivery in your dashboard separately using non-sensitive content/session labels. Otherwise leave it disabled.
6. Start `.\demo.ps1 chat --live-trace`. Optional repeatable label: `--session capstone-demo`. The default profile is `prima`, post-turn trace display is `summary`, and memory is in process only.

Git Bash equivalents are `./demo preflight` and `./demo chat --live-trace`. Both wrappers use the active environment's `python` and accept the same CLI flags.

Live trace shows actual workflow events emitted while the turn executes. `/live on` and `/live off` control subsequent turns; `/live` shows the setting. The normal `/trace` command controls the completed post-turn diagnostic summary. `[OK]` marks phase completion, not a guaranteed successful answer. Missing live values/skipped phases are omitted; retries remain visible. Live output is bounded to 50 events per turn and excludes background maintenance and raw payloads. The default without `--live-trace` remains quiet. No real-provider/E2E validation is implied by this display.

## During the demo

1. Suggested first prompt: “I have my capstone demo today. I'm pretty nervous because the last presentation didn't go as well as I wanted.”
2. Inspect `/state` and `/memory` for actual affect/state and stored memories. Some fields may be unavailable; do not claim guaranteed memory admission.
3. Suggested second prompt: “I tend to rush when I'm nervous. Give me three things I should focus on right before I start.”
4. Inspect `/plan` and `/world`. World effects are forecasts; constraint predictions are not confirmed violations.
5. Suggested continuity prompt: “What did I tell you earlier was worrying me about today's demo?” Generated answers are not guaranteed.
6. Use `/trace full` for bounded diagnostics, `/trace summary` to return to the compact display. Reflection is selective and only runs when uncertainty/control logic requires it; “skipped” is valid.
7. Optional architecture comparison: `/profile model`, ask a prompt, then `/profile prima`, ask an equivalent prompt. Provider/model remain unchanged, conversation state remains, and wording is stochastic. Compare executed architecture, not subjective answer quality. `/reset` starts a fresh conversation while preserving profile/trace/live settings.
8. Finish with `/quit`. EOF and Ctrl+C also run cleanup.

## Fallbacks and limits

- Groq unavailable: exit, start the local Ollama server and use `.\demo.ps1 preflight --strict --provider ollama --model "EXACT-INSTALLED-ID"`, then `.\demo.ps1 chat --provider ollama --model "EXACT-INSTALLED-ID"`. No model downloads or provider changes inside a session.
- Langfuse unavailable: continue chatting. Install/fix optional dependencies and credentials before the next run; successful chat does not prove export delivery.
- Telemetry shutdown waits at most three seconds and may drop queued traces. Warnings do not fail the demo.
- An in-flight HTTP worker may continue until its configured timeout after Ctrl+C. Wait for cleanup rather than repeatedly interrupting. Idle console input can delay cold maintenance.
- `/clear` changes presentation only; `/reset` clears the active conversation. In normal mode, exiting loses state and memory.

## Optional persistence verification

1. Run `.\demo.ps1 preflight --persist --session persistence-check`. Use `--strict` to verify opening Chroma and provider connectivity.
2. Start `.\demo.ps1 chat --persist --session persistence-check` and enter one or two turns.
3. Inspect `/state` and `/memory`, then `/quit`.
4. Restart the same command/session ID. Confirm the resume message, state version, affect/PAD and stored memories before another turn.
5. Ask a continuity question. The next successful turn should advance the persisted state version.
6. `/plan` and `/world` describe turns from the current process; the rolling affect history window starts fresh. Persisted emotional state and momentum still seed evolution.
7. `/reset` clears only this session's persisted state, memories and graph, then moves to a displayed new session ID. It preserves persistence and all active configuration. Old session IDs reopen empty. A partial reset error ends the session; inspect storage before reopening it.

Data lives under ignored `.demo_data/` by default. Configure `PRIMA_DEMO_DATA_DIR` or `--data-dir PATH` if needed and keep custom storage out of Git. Use one writer per session. Reuse the embedding configuration across launches; generation provider/model can change independently. Persistence is local and unencrypted. No sessions are automatically selected or migrated.
