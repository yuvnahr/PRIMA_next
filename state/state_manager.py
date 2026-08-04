"""Cognitive-state ownership with optimistic version checks."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from threading import RLock
from typing import Any

from state.cognitive_state import CognitiveState, utc_now


class StateVersionConflict(RuntimeError):
    """A stale write attempted to replace newer state."""


class StateManager(ABC):
    """State ownership boundary."""

    @abstractmethod
    def load(self, session_id: str) -> CognitiveState: ...
    @abstractmethod
    def apply_delta(self, session_id: str, changes: Mapping[str, Any], expected_version: int) -> CognitiveState: ...
    @abstractmethod
    def save(self, session_id: str, state: CognitiveState, expected_version: int) -> CognitiveState: ...
    @abstractmethod
    def reset(self, session_id: str | None = None) -> None: ...


class InMemoryStateManager(StateManager):
    """Deterministic process-local state manager."""

    def __init__(self) -> None:
        self._states: dict[str, CognitiveState] = {}
        self._lock = RLock()

    def load(self, session_id: str) -> CognitiveState:
        with self._lock:
            return CognitiveState.from_dict(self._states.get(session_id, CognitiveState()).to_dict())

    def apply_delta(self, session_id: str, changes: Mapping[str, Any], expected_version: int) -> CognitiveState:
        state = self.load(session_id)
        for section, values in changes.items():
            target = getattr(state, section, None)
            if isinstance(target, MutableMapping) and isinstance(values, Mapping):
                target.update(values)
            else:
                raise ValueError(f"Unsupported state delta section: {section}")
        return self.save(session_id, state, expected_version)

    def save(self, session_id: str, state: CognitiveState, expected_version: int) -> CognitiveState:
        if not session_id:
            raise ValueError("session_id must not be empty")
        with self._lock:
            current = self._states.get(session_id)
            actual = current.version if current else 0
            if actual != expected_version:
                raise StateVersionConflict(
                    f"State version conflict for {session_id!r}: expected {expected_version}, found {actual}."
                )
            saved = CognitiveState.from_dict(state.to_dict())
            saved.version, saved.updated_at = expected_version + 1, utc_now()
            saved.created_at = current.created_at if current else state.created_at
            self._states[session_id] = saved
            return CognitiveState.from_dict(saved.to_dict())

    def reset(self, session_id: str | None = None) -> None:
        with self._lock:
            if session_id is None:
                self._states.clear()
            else:
                self._states.pop(session_id, None)


class JsonStateManager(InMemoryStateManager):
    """Configured persistent state manager backed by versioned JSON."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "1.0":
                raise ValueError("Unsupported cognitive state schema version.")
            self._states = {key: CognitiveState.from_dict(value) for key, value in payload.get("sessions", {}).items()}

    def save(self, session_id: str, state: CognitiveState, expected_version: int) -> CognitiveState:
        # ponytail: process-local lock; use database transactions when multi-process state writers are required.
        with self._lock:
            if self.path.exists():
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                self._states = {key: CognitiveState.from_dict(value) for key, value in payload.get("sessions", {}).items()}
            saved = super().save(session_id, state, expected_version)
            self._write()
            return saved

    def reset(self, session_id: str | None = None) -> None:
        super().reset(session_id)
        self._write()

    def _write(self) -> None:
        payload = {"schema_version": "1.0", "sessions": {key: value.to_dict() for key, value in self._states.items()}}
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)
