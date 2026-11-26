"""Translate failures into fixed messages; never echo arbitrary exception text."""

from __future__ import annotations

import re

from demo_cli.storage import PersistenceError


def safe_reason(error: BaseException | str, *, provider: str = "") -> str:
    if isinstance(error, PersistenceError):
        return str(error)
    text = str(error)
    lower = text.lower()
    if "api key is not configured" in lower and provider in {"groq", "openrouter", "openai", "anthropic"}:
        return f"API key is not configured. Set {provider.upper()}_API_KEY."
    if "authentication" in lower or re.search(r"\b(?:401|403)\b", text):
        return "authentication failed; check the configured API key."
    if "timeout" in lower or "timed out" in lower:
        return "provider request timed out; check connectivity and try again."
    if re.search(r"\b404\b", text):
        return ("Ollama model was not found; check the exact installed model ID."
                if provider in {"ollama", "local"} else "provider endpoint or model was not found.")
    if "429" in text or "rate limit" in lower:
        return "provider rate limit reached; wait and try again."
    if "connectionerror" in lower or "urlerror" in lower or "connection refused" in lower:
        return "Ollama is unavailable; start the local server." if provider in {"ollama", "local"} else "provider is unreachable; check connectivity."
    if "http request failed" in lower:
        return "provider HTTP request failed; check service availability."
    if "unknown provider" in lower:
        return "Unknown provider. Select a supported provider such as groq, openrouter or ollama."
    # Only these application-owned configuration messages may pass through.
    if text in {
        "No provider configured. Set DEFAULT_PROVIDER or use --provider groq or --provider ollama.",
        "Ollama requires --model <exact-local-model-id> or a configured DEFAULT_MODEL.",
        "Generation model and provider must be configured.",
        "Profile must be prima, rag or model.", "Trace must be off, summary or full.",
        "Session ID must not be empty.", "Temperature must be numeric and max-tokens must be an integer.",
        "Demo data directory must not be empty.",
        "Unable to resolve demo data directory. Check PRIMA_DEMO_DATA_DIR or --data-dir.",
        "temperature must be between 0 and 2",
        "Token limit and timeout must be positive; retries cannot be negative.",
        "--trace-content requires --langfuse or LANGFUSE_ENABLED=true.",
        "Comparison requires --profile prima; the other branch is a direct model chat.",
        "Comparison uses process-local histories. Omit --persist; reset starts both branches fresh.",
        "Comparison requires process-local storage and the prima profile.",
        "Comparison requires an OpenAI-compatible provider with role-based history (e.g. Groq).",
        "The branches have uneven histories after a failure. Use /reset before continuing.",
        "Enter a nonempty comparison prompt.",
        "Baseline history limit reached; no messages were dropped and neither branch ran. Use /reset.",
        "The base model returned an empty answer.",
    }:
        return text
    return "operation did not complete; check configuration or use the existing diagnostic logs."
