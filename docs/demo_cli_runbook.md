# Demo CLI runbook

## Before the demo

1. Open the repository in PowerShell and activate the intended environment: `. .\.venv\Scripts\Activate.ps1`.
2. Set provider/model and credentials in the environment or ignored `.env`; keep keys off screen.
3. Run `.\demo.ps1 preflight`. Fix required configuration errors. Optional Langfuse warnings permit chat.
4. Run `.\demo.ps1 preflight --strict` to verify the selected provider with one tiny generation. This can use hosted-provider tokens. For Ollama, use an exact installed model ID; do not download models on stage.
5. If using Langfuse, install `requirements-observability.txt`, configure its three credentials/URL variables, and enable `LANGFUSE_ENABLED`. Preflight checks local configuration/dependencies only; verify delivery in your dashboard separately using non-sensitive content/session labels. Otherwise leave it disabled.
6. Start `.\demo.ps1 chat`. Optional repeatable label: `--session capstone-demo`. The default profile is `prima`, trace display is `summary`, and memory is in process only.

Git Bash equivalents are `./demo preflight` and `./demo chat`. Both wrappers use the active environment's `python` and accept the same CLI flags.

## During the demo

1. Suggested first prompt: “I have my capstone demo today. I'm pretty nervous because the last presentation didn't go as well as I wanted.”
2. Inspect `/state` and `/memory` for actual affect/state and stored memories. Some fields may be unavailable; do not claim guaranteed memory admission.
3. Suggested second prompt: “I tend to rush when I'm nervous. Give me three things I should focus on right before I start.”
4. Inspect `/plan` and `/world`. World effects are forecasts; constraint predictions are not confirmed violations.
5. Suggested continuity prompt: “What did I tell you earlier was worrying me about today's demo?” Generated answers are not guaranteed.
6. Use `/trace full` for bounded diagnostics, `/trace summary` to return to the compact display. Reflection is selective and only runs when uncertainty/control logic requires it; “skipped” is valid.
7. Optional architecture comparison: `/profile model`, ask a prompt, then `/profile prima`, ask an equivalent prompt. Provider/model remain unchanged, conversation state remains, and wording is stochastic. Compare executed architecture, not subjective answer quality. `/reset` starts a fresh conversation while preserving profile/trace settings.
8. Finish with `/quit`. EOF and Ctrl+C also run cleanup.

## Fallbacks and limits

- Groq unavailable: exit, start the local Ollama server and use `.\demo.ps1 preflight --strict --provider ollama --model "EXACT-INSTALLED-ID"`, then `.\demo.ps1 chat --provider ollama --model "EXACT-INSTALLED-ID"`. No model downloads or provider changes inside a session.
- Langfuse unavailable: continue chatting. Install/fix optional dependencies and credentials before the next run; successful chat does not prove export delivery.
- Telemetry shutdown waits at most three seconds and may drop queued traces. Warnings do not fail the demo.
- An in-flight HTTP worker may continue until its configured timeout after Ctrl+C. Wait for cleanup rather than repeatedly interrupting. Idle console input can delay cold maintenance.
- `/clear` changes presentation only; `/reset` clears this in-memory conversation. Exiting loses state and memory. Persistence is separate work.
