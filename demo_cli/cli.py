"""Run with ``python -m demo_cli.cli chat``, ``status`` or ``preflight``."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from demo_cli.comparison import ComparisonSession

try:
    from config.settings import get_settings
    from demo_cli.commands import HELP, dispatch_command
    from demo_cli.config import PROFILES, TraceMode, resolve_config
    from demo_cli.errors import safe_reason
    from demo_cli.live_eval import EvalUnavailable
    from demo_cli.preflight import preflight
    from demo_cli.renderer import Renderer
    from demo_cli.session import DemoSession
    from demo_cli.storage import PersistenceError
    from llm.provider import ProviderError
    from monitoring.tracing import shutdown_tracing
    from runtime.contracts import ExecutionStatus
except Exception:
    if __name__ != "__main__":
        raise
    print("PRIMA DEMO PREFLIGHT\nPRIMA runtime / Demo CLI imports failed.\nResult: NOT READY")
    print("Fix: activate your Python environment and install requirements-core.txt.")
    raise SystemExit(1) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat through the canonical PRIMA runtime.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("chat", "compare", "status", "preflight"):
        command = commands.add_parser(name)
        if name in {"chat", "compare", "status", "preflight"}:
            command.add_argument("--langfuse", action="store_true", help="Enable optional Langfuse tracing for this process.")
            command.add_argument("--trace-content", action="store_true", help="Export redacted prompt, answer and phase content to Langfuse (opt in).")
        if name in {"chat", "compare"}:
            command.add_argument("--live-trace", action="store_true", help="Display actual foreground workflow events during each turn.")
        if name == "preflight":
            command.add_argument("--strict", action="store_true", help="Make one small live provider request (may use tokens).")
        if name == "compare":
            command.add_argument("--judge-model", help="Model for /eval on the same provider (Groq defaults to GPT-OSS 20B).")
        command.add_argument("--provider", help="Provider name (e.g. groq or ollama); defaults to project settings.")
        command.add_argument("--model", help="Exact model ID; required for Ollama unless DEFAULT_MODEL is configured.")
        command.add_argument("--profile", choices=tuple(PROFILES))
        command.add_argument("--session", help="Stable ID for this process-local conversation.")
        command.add_argument("--trace", choices=tuple(mode.value for mode in TraceMode))
        command.add_argument("--temperature", type=float)
        command.add_argument("--max-tokens", type=int)
        command.add_argument("--no-color", action="store_true")
        command.add_argument("--persist", action="store_true", help="Opt into session-scoped persistent state and memory.")
        command.add_argument("--data-dir", help="Demo persistence root; defaults to PRIMA_DEMO_DATA_DIR or .demo_data.")
    return parser


async def chat(session: DemoSession, renderer: Renderer) -> None:
    try:
        session.observe_workflow(renderer)
        await session.start()
        renderer.rows("PRIMA-NEXT Interactive Demo", (row for row in session.status_rows() if row[0] not in {"Memory", "Data root"}))
        for message in session.persistence_messages():
            renderer.message(message)
        if session.observability.warning:
            renderer.message(f"Warning: {session.observability.warning}")
        renderer.message("Type /help for commands.")
        while True:
            try:
                # shortcut: stdin blocks idle maintenance; use async input if idle/background UI is added.
                text = input("You > ")
            except (EOFError, KeyboardInterrupt):
                renderer.message("\nSession ended.")
                break
            if not text.strip():
                continue
            if text.lstrip().startswith("/"):
                try:
                    if not await dispatch_command(text, session, renderer):
                        break
                except PersistenceError:
                    raise
                except Exception:
                    renderer.error("The command did not complete.", provider=session.config.generation.provider)
                continue
            try:
                response = await session.send(text)
            except (KeyboardInterrupt, asyncio.CancelledError):
                renderer.message("Interrupted. Shutting down; an active provider request may run until its timeout.")
                raise
            except PersistenceError:
                raise
            except Exception as exc:
                renderer.error(safe_reason(exc, provider=session.config.generation.provider), provider=session.config.generation.provider)
                continue
            diagnostics = session.latest_diagnostics
            if response.status in {ExecutionStatus.FAILED, ExecutionStatus.REJECTED, ExecutionStatus.CANCELLED}:
                renderer.error(
                    safe_reason("; ".join(diagnostics.errors) if diagnostics and diagnostics.errors else response.status.value,
                                provider=session.config.generation.provider),
                    provider=session.config.generation.provider, stage=diagnostics.failed_stage if diagnostics else None,
                )
            elif response.output_text is not None:
                renderer.answer(response.output_text)
            else:
                renderer.message(f"PRIMA > {response.outcome.value if response.outcome else response.status.value}")
            if diagnostics is not None:
                renderer.trace(diagnostics, session.trace_mode)
            await asyncio.sleep(0)
    finally:
        try:
            await session.close()
        except Exception:
            renderer.message("Warning: runtime cleanup did not complete; session is ending.")


async def compare(session: ComparisonSession, renderer: Renderer) -> None:
    try:
        session.prima.observe_workflow(renderer)
        await session.prima.start()
        renderer.rows("PRIMA / BASE MODEL COMPARISON", session.status_rows())
        renderer.message("Same model/settings; separate histories. Technical completion does not assess answer quality.")
        renderer.message("/eval grades the latest pair. /help lists workflow inspections and other commands.")
        if session.prima.observability.warning:
            renderer.message(f"Warning: {session.prima.observability.warning}")
        while True:
            try:
                text = input("Both > ")
            except (EOFError, KeyboardInterrupt):
                break
            if not text.strip():
                continue
            try:
                if text.lstrip().startswith("/"):
                    command = text.strip().lower()
                    if command in {"/quit", "/exit"}:
                        break
                    if command == "/reset":
                        await session.reset()
                        renderer.rows("BOTH BRANCHES RESET", session.status_rows())
                    elif command == "/status":
                        renderer.rows("COMPARISON STATUS", session.status_rows())
                    elif command == "/help":
                        renderer.message("\n".join(line for line in HELP.splitlines() if not line.startswith("/profile")))
                        renderer.message("/eval                    Grade the latest paired answers with one judge request")
                        renderer.message("Workflow inspections describe PRIMA. /reset clears both branches.")
                    elif command.split()[0] == "/eval":
                        if command != "/eval":
                            renderer.message("Usage: /eval")
                        else:
                            try:
                                evaluation = await session.evaluate_latest(renderer.message)
                            except EvalUnavailable as exc:
                                renderer.message(f"Evaluation unavailable. {exc}", error=True)
                            else:
                                renderer.rows("QUICK RUBRIC ESTIMATE", evaluation.rows())
                    elif command.split()[0] == "/profile":
                        renderer.message("Comparison profiles are fixed: ordinary base chat versus prima.")
                    else:
                        await dispatch_command(text, session.prima, renderer)
                    continue
                turn = await session.send(text)
                renderer.comparison(turn)
                if session.needs_reset:
                    renderer.message("A branch failed; histories may differ. The successful answer is retained. Use /reset before continuing.")
                if session.prima.latest_diagnostics:
                    renderer.trace(session.prima.latest_diagnostics, session.prima.trace_mode)
            except Exception as exc:
                renderer.error(safe_reason(exc, provider=session.config.generation.provider), provider=session.config.generation.provider)
            await asyncio.sleep(0)
    finally:
        await session.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    renderer = Renderer(no_color=args.no_color)
    try:
        settings = get_settings()
    except Exception:
        if args.command == "preflight":
            renderer.rows("PRIMA DEMO PREFLIGHT", (("Configuration", "unavailable"), ("Result", "NOT READY")))
        renderer.error("Unable to load project settings. Check your .env configuration.")
        return 1
    try:
        if args.command == "preflight":
            return preflight(args, settings, renderer)
        config = resolve_config(args, settings)
        renderer = Renderer(no_color=config.no_color)
        if args.command == "compare":
            from demo_cli.comparison import ComparisonSession

            asyncio.run(compare(ComparisonSession(config, settings=settings), renderer))
            return 0
        session = DemoSession(config, settings=settings)
        if args.command == "status":
            try:
                renderer.rows("PRIMA STATUS", session.status_rows())
            finally:
                asyncio.run(session.close())
        else:
            asyncio.run(chat(session, renderer))
        return 0
    except (KeyboardInterrupt, asyncio.CancelledError):
        renderer.message("Session ended.")
        return 0
    except (ValueError, ProviderError) as exc:
        provider = (args.provider or settings.default_provider).strip().lower()
        renderer.error(safe_reason(exc, provider=provider), provider=provider)
        return 1
    except Exception:
        renderer.error("Unable to initialize the runtime. Check configuration and installed requirements.")
        return 1
    finally:
        try:
            shutdown_tracing()
        except Exception:
            renderer.message("Warning: telemetry shutdown failed; queued traces may be lost.")


if __name__ == "__main__":
    raise SystemExit(main())
