"""Run with ``python -m demo_cli.cli chat``, ``status`` or ``preflight``."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

try:
    from config.settings import get_settings
    from demo_cli.commands import dispatch_command
    from demo_cli.config import PROFILES, TraceMode, resolve_config
    from demo_cli.errors import safe_reason
    from demo_cli.preflight import preflight
    from demo_cli.renderer import Renderer
    from demo_cli.session import DemoSession
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
    for name in ("chat", "status", "preflight"):
        command = commands.add_parser(name)
        if name in {"chat", "preflight"}:
            command.add_argument("--langfuse", action="store_true", help="Enable optional Langfuse tracing for this process.")
        if name == "preflight":
            command.add_argument("--strict", action="store_true", help="Make one small live provider request (may use tokens).")
        command.add_argument("--provider", help="Provider name (e.g. groq or ollama); defaults to project settings.")
        command.add_argument("--model", help="Exact model ID; required for Ollama unless DEFAULT_MODEL is configured.")
        command.add_argument("--profile", choices=tuple(PROFILES))
        command.add_argument("--session", help="Stable ID for this process-local conversation.")
        command.add_argument("--trace", choices=tuple(mode.value for mode in TraceMode))
        command.add_argument("--temperature", type=float)
        command.add_argument("--max-tokens", type=int)
        command.add_argument("--no-color", action="store_true")
    return parser


async def chat(session: DemoSession, renderer: Renderer) -> None:
    try:
        await session.start()
        renderer.rows("PRIMA-NEXT Interactive Demo", (row for row in session.status_rows() if row[0] != "Memory"))
        if session.observability.warning:
            renderer.message(f"Warning: {session.observability.warning}")
        renderer.message("Type /help for commands.")
        while True:
            try:
                # shortcut: stdin blocks idle maintenance; use cancellable async input when live events are added.
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
                except Exception:
                    renderer.error("The command did not complete.", provider=session.config.generation.provider)
                continue
            try:
                response = await session.send(text)
            except (KeyboardInterrupt, asyncio.CancelledError):
                renderer.message("Interrupted. Shutting down; an active provider request may run until its timeout.")
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
        session = DemoSession(config, settings=settings)
        if args.command == "status":
            renderer.rows("PRIMA STATUS", session.status_rows())
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
