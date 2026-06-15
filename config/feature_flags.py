"""Feature flag helpers backed by environment variables.

Use `FEATURE_<NAME>=1` to enable a flag.
"""
import os


def is_enabled(name: str) -> bool:
    v = os.getenv(f"FEATURE_{name.upper()}", "false").lower()
    return v in ("1", "true", "yes", "on")
