"""Simple permission policy for controlling tool access.

This is intentionally minimal. A production system should integrate with the
application's identity and policy systems (RBAC, ABAC) and include audit
callbacks.
"""
from collections.abc import Iterable


class PermissionPolicy:
    def __init__(self, allowed_tools: Iterable[str] | None = None):
        self.allowed = set(allowed_tools or [])

    def is_allowed(self, tool_name: str) -> bool:
        return tool_name in self.allowed

    def allow(self, tool_name: str):
        self.allowed.add(tool_name)

    def revoke(self, tool_name: str):
        self.allowed.discard(tool_name)
