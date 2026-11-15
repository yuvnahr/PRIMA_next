"""Non-destructive construction checks, with one opt-in provider request."""

from __future__ import annotations

import asyncio
import sys
from argparse import Namespace

from config.settings import Settings
from demo_cli.config import resolve_config
from demo_cli.errors import safe_reason
from demo_cli.renderer import Renderer
from demo_cli.session import DemoSession
from monitoring.tracing import Tracing, langfuse_readiness


def preflight(args: Namespace, settings: Settings, renderer: Renderer) -> int:
    rows = [("Python", f"OK ({sys.version_info.major}.{sys.version_info.minor})"), ("Demo CLI", "OK")]
    fixes = []
    warnings = []
    session = None
    provider = args.provider or settings.default_provider
    try:
        config = resolve_config(args, settings)
        provider = config.generation.provider
        rows.extend((("Configuration", "OK"), ("Provider", provider), ("Model", config.generation.model),
                     ("Profile", config.profile.value)))
        if provider in {"groq", "openai", "anthropic"}:
            present = bool((getattr(settings, f"{provider}_api_key", None) or "").strip())
            rows.append((f"{provider.title()} key", "configured" if present else "missing"))
            if not present:
                fixes.append(f"Set {provider.upper()}_API_KEY in the configured environment.")
        # Inspect optional SDK/config without creating exporters or emitting traces.
        status, hint = langfuse_readiness(settings, enabled=config.langfuse)
        rows.append(("Langfuse", status if status in {"disabled", "ready"} else f"{status} (optional)"))
        if hint:
            warnings.append(hint)
        session = DemoSession(config, settings=settings, observability=Tracing())
        rows.extend((("PRIMA runtime", "OK"), ("Provider setup", "OK"),
                     ("Memory", session.runtime.repository_selection.backend)))
        if args.strict and not fixes:
            session.runtime.llm_client.chat(
                "Reply OK.", generation_config=config.generation.with_overrides(
                    max_output_tokens=8, timeout_seconds=10, retries=0,
                ),
            )
            rows.append(("Provider live", "OK"))
        else:
            rows.append(("Provider live", "not checked (use --strict)"))
    except Exception as exc:
        fixes.append(safe_reason(exc, provider=provider))
    finally:
        if session is not None:
            try:
                asyncio.run(session.close())
            except Exception:
                fixes.append("Runtime cleanup failed; check the local runtime configuration.")
    result = "NOT READY" if fixes else "READY WITH WARNING" if warnings else "READY"
    renderer.rows("PRIMA DEMO PREFLIGHT", (*rows, ("Result", result)))
    for fix in fixes:
        renderer.message(f"Fix: {fix}")
    for warning in warnings:
        renderer.message(f"Warning: {warning}")
    return 1 if fixes else 0
