"""Dependency injection interfaces for reflection."""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Return a deterministic or model-backed generation."""


class Verifier(Protocol):
    def verify(self, proposal: str, observation: str, context: object) -> object:
        """Verify a proposal/observation pair."""
