"""Lightweight secret manager interface.

This is a small abstraction over environment-backed secrets. It intentionally
does not persist secrets in code or on disk. Integrate with a real secret
backend (Vault, AWS Secrets Manager, GCP Secret Manager) by implementing the
`backend_get` hook or by subclassing `SecretManager`.
"""
import os
from typing import Optional, Callable


class SecretManager:
    def __init__(self, backend_get: Optional[Callable[[str], Optional[str]]] = None):
        """backend_get is a callable(name) -> Optional[str] used to fetch secrets.

        If omitted, environment variables are used.
        """
        self.backend_get = backend_get

    def get(self, name: str) -> Optional[str]:
        if self.backend_get:
            v = self.backend_get(name)
            if v:
                return v
        return os.getenv(name)
