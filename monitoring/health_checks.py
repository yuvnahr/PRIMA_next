"""Simple health checks for observability dashboarding.

These checks are lightweight and intended to be called by an orchestration
service or HTTP health endpoint.
"""
import logging
import socket

try:
    import requests
except Exception:
    requests = None  # type: ignore

from typing import Any

# default then attempt to load real settings provider
def get_settings() -> Any:
    return None
try:
    from config import settings as _settings
    get_settings = _settings.get_settings
except Exception as exc:
    logging.getLogger(__name__).debug(
        "config.settings unavailable; using default settings stub: %s", exc
    )


def check_health() -> dict[str, str]:
    status = {"ok": "true"}
    settings = get_settings()
    # Check that the configured Ollama URL is reachable (if available)
    if settings is not None and getattr(settings, "ollama_url", None):
        url = settings.ollama_url
        if requests is None:
            status["ollama"] = "skipped (requests not installed)"
        else:
            try:
                r = requests.get(url, timeout=2)
                status["ollama"] = f"reachable ({r.status_code})"
            except Exception as e:
                status["ollama"] = f"unreachable ({e})"
    # Simple DNS check for outbound network
    try:
        socket.gethostbyname("example.com")
        status["network"] = "ok"
    except Exception:
        status["network"] = "fail"
    return status
