"""Application settings using pydantic-settings when available.

This module provides a single `get_settings()` function that returns a
singleton Settings object. By default values are read from environment
variables; sensitive values (API keys) should not be checked into source.
"""

try:
    from pydantic_settings import BaseSettings
except Exception:  # pragma: no cover - fall back to pydantic if pydantic-settings isn't installed
    try:
        from pydantic import BaseSettings
    except Exception:
        BaseSettings = object  # type: ignore


class Settings(BaseSettings):
    default_provider: str = "openai"
    default_model: str = "gpt-4o"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_url: str = "http://localhost:11434"
    rate_limit_per_minute: int = 60

    class Config:  # pydantic v1 compatibility
        env_file = ".env"
        env_prefix = ""


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings()
    return _SETTINGS
