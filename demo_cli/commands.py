"""Session commands and bounded inspection, independent of request execution."""

from __future__ import annotations

from demo_cli.config import PROFILES, TraceMode
from demo_cli.renderer import Renderer
from demo_cli.session import DemoSession

HELP = """COMMANDS
/help                    Show commands
/status                  Show current runtime configuration
/trace off|summary|full   Change trace display
/live [on|off]           Show or change live workflow tracing
/state                   Show current cognitive state
/memory                  Show up to five stored memories
/plan                    Show the last turn's plan
/world                   Show the last turn's world simulation
/profile prima|rag|model Change the next turn's profile
/clear                   Clear terminal output only
/reset                   Start a fresh conversation
/quit                    Exit (/exit also works)"""


async def dispatch_command(text: str, session: DemoSession, renderer: Renderer) -> bool:
    """Return false only when a valid quit command ends the session."""
    parts = text.strip().split()
    if not parts:
        return True
    command, args = parts[0].lower(), parts[1:]
    if command in {"/quit", "/exit"} and not args:
        return False
    inspections = {
        "/state": ("COGNITIVE STATE", session.current_state),
        "/memory": ("MEMORY", session.stored_memories),
        "/plan": ("CURRENT PLAN", session.latest_plan),
        "/world": ("WORLD MODEL", session.latest_world),
    }
    if command in inspections:
        if args:
            renderer.message(f"Usage: {command}")
        else:
            title, inspect = inspections[command]
            renderer.rows(title, inspect())
    elif command == "/clear":
        if args:
            renderer.message("Usage: /clear")
        else:
            renderer.clear()
    elif command == "/reset":
        if args:
            renderer.message("Usage: /reset")
        else:
            previous, current = await session.reset()
            renderer.rows("SESSION RESET", (("Previous session", previous), ("New session", current),
                                             ("State", "fresh"), ("Memory", "empty")))
            for message in session.persistence_messages():
                renderer.message(message)
    elif command == "/profile":
        if not args:
            renderer.rows("PROFILE", (("Current", session.current_profile.value), ("Available", ", ".join(PROFILES))))
        elif len(args) != 1:
            renderer.message("Usage: /profile prima|rag|model")
        else:
            try:
                previous_profile = session.set_profile(args[0])
            except ValueError as exc:
                renderer.message(str(exc))
            else:
                renderer.message(f"Profile changed: {previous_profile.value} -> {session.current_profile.value}")
    elif command == "/help" and not args:
        renderer.message(HELP)
    elif command == "/status" and not args:
        renderer.rows("PRIMA STATUS", session.status_rows())
    elif command == "/live":
        if len(args) > 1 or (args and args[0] not in {"on", "off"}):
            renderer.message("Usage: /live [on|off]")
        else:
            if args:
                session.live_trace = args[0] == "on"
            renderer.message(f"Live workflow trace: {'on' if session.live_trace else 'off'}")
    elif command == "/trace":
        if not args:
            renderer.message(f"Trace: {session.trace_mode.value}. Choices: off, summary, full.")
        elif len(args) == 1 and args[0] in {mode.value for mode in TraceMode}:
            session.trace_mode = TraceMode(args[0])
            renderer.message(f"Trace: {session.trace_mode.value}")
        else:
            renderer.message("Unknown trace mode or arguments. Usage: /trace off|summary|full. Available: off, summary, full.")
    else:
        renderer.message("Unknown command or arguments. Type /help for commands.")
    return True
