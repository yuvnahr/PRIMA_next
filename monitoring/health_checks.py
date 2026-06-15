"""Simple health checks for observability dashboarding.

These checks are lightweight and intended to be called by an orchestration
service or HTTP health endpoint.
"""
from typing import Dict
import socket

try:
    import requests
except Exception:
    requests = None  # type: ignore

try:
    from config.settings import get_settings
except Exception:
    def get_settings():
        return None


def check_health() -> Dict[str, str]:
    status = {"ok": "true"}
    settings = get_settings()
    # Check that the configured Ollama URL is reachable (if available)
    if settings is not None and getattr(settings, "ollama_url", None):
        url = getattr(settings, "ollama_url")
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
