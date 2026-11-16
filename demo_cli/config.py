"""Resolve only the settings needed by one CLI conversation."""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from uuid import uuid4

from config.settings import Settings, get_settings
from llm.generation_config import GenerationConfig
from runtime.contracts import ExecutionProfile

PROFILES = {
    "prima": ExecutionProfile.PRIMA_FULL,
    "rag": ExecutionProfile.SIMPLE_RAG,
    "model": ExecutionProfile.MODEL_ONLY,
}


class TraceMode(str, Enum):
    OFF = "off"
    SUMMARY = "summary"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class DemoConfig:
    generation: GenerationConfig
    profile: ExecutionProfile
    session_id: str
    trace_mode: TraceMode
    no_color: bool
    langfuse: bool = False
    persist: bool = False
    data_dir: Path = Path(__file__).resolve().parents[1] / ".demo_data"
    live_trace: bool = False


def resolve_config(
    args: Namespace, settings: Settings | None = None, environment: Mapping[str, str] | None = None
) -> DemoConfig:
    """CLI flags override project/provider settings and small demo defaults."""
    settings = settings or get_settings()
    env = os.environ if environment is None else environment
    provider = (args.provider if args.provider is not None else settings.default_provider).strip().lower()
    if not provider:
        raise ValueError("No provider configured. Set DEFAULT_PROVIDER or use --provider groq or --provider ollama.")
    model = args.model
    if model is None:
        if "default_model" in getattr(settings, "model_fields_set", set()):
            model = settings.default_model
        elif provider == "groq":
            model = "openai/gpt-oss-120b"
        elif provider in {"ollama", "local"}:
            model = ""
        else:
            model = settings.default_model
    if provider in {"ollama", "local"} and not model.strip():
        raise ValueError("Ollama requires --model <exact-local-model-id> or a configured DEFAULT_MODEL.")
    profile_name = args.profile if args.profile is not None else env.get("PRIMA_DEMO_PROFILE", "prima")
    if profile_name not in PROFILES:
        raise ValueError("Profile must be prima, rag or model.")
    trace = args.trace if args.trace is not None else env.get("PRIMA_DEMO_TRACE", "summary")
    try:
        trace_mode = TraceMode(trace)
    except ValueError:
        raise ValueError("Trace must be off, summary or full.") from None
    session_id = args.session if args.session is not None else env.get("PRIMA_DEMO_SESSION")
    if session_id is None:
        session_id = f"demo-{uuid4().hex[:12]}"
    if not session_id.strip():
        raise ValueError("Session ID must not be empty.")
    try:
        temperature = float(args.temperature if args.temperature is not None else env.get("PRIMA_DEMO_TEMPERATURE", "0"))
        max_tokens = int(args.max_tokens if args.max_tokens is not None else env.get("PRIMA_DEMO_MAX_TOKENS", "4096"))
    except ValueError:
        raise ValueError("Temperature must be numeric and max-tokens must be an integer.") from None
    directory = getattr(args, "data_dir", None)
    if directory is None:
        directory = env.get("PRIMA_DEMO_DATA_DIR", settings.prima_demo_data_dir)
    if not directory.strip():
        raise ValueError("Demo data directory must not be empty.")
    try:
        data_dir = Path(directory).expanduser()
        if not data_dir.is_absolute():
            data_dir = Path(__file__).resolve().parents[1] / data_dir
        data_dir = data_dir.resolve()
    except (OSError, ValueError, RuntimeError):
        raise ValueError("Unable to resolve demo data directory. Check PRIMA_DEMO_DATA_DIR or --data-dir.") from None
    return DemoConfig(
        GenerationConfig(model=model.strip(), provider=provider, temperature=temperature, max_output_tokens=max_tokens),
        PROFILES[profile_name], session_id, trace_mode, args.no_color or "NO_COLOR" in env,
        getattr(args, "langfuse", False) or settings.langfuse_enabled,
        persist=getattr(args, "persist", False), data_dir=data_dir,
        live_trace=getattr(args, "live_trace", False),
    )
