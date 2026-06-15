"""Helpers to detect environment (dev/staging/prod)."""
import os


def get_environment() -> str:
    return os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()


def is_production() -> bool:
    return get_environment() in ("prod", "production")
