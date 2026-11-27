"""Cognitive-state ownership with optimistic version checks."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from tempfile import NamedTemporaryFile
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
    def apply_delta(
        self, session_id: str, changes: Mapping[str, Any], expected_version: int, *,
        expected_generation: tuple[int, int] = (0, 0),
    ) -> CognitiveState: ...
    @abstractmethod
    def save(self, session_id: str, state: CognitiveState, expected_version: int) -> CognitiveState: ...
    @abstractmethod
    def reset(self, session_id: str | None = None) -> None: ...


class InMemoryStateManager(StateManager):
    """Deterministic process-local state manager."""

    def __init__(self) -> None:
        self._states: dict[str, CognitiveState] = {}
        self._epoch = 0
        self._resets: dict[str, int] = {}
        self._lock = RLock()

    def load(self, session_id: str) -> CognitiveState:
        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                return CognitiveState(generation=(self._epoch, self._resets.get(session_id, 0)))
            return CognitiveState.from_dict(state.to_dict())

    def apply_delta(
        self, session_id: str, changes: Mapping[str, Any], expected_version: int, *,
        expected_generation: tuple[int, int] = (0, 0),
    ) -> CognitiveState:
        state = self.load(session_id)
        if state.generation != expected_generation:
            raise StateVersionConflict("Delta is from an earlier reset generation; load a fresh snapshot.")
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
            states, epoch, resets = self._committed()
            if state.generation != (epoch, resets.get(session_id, 0)):
                raise StateVersionConflict(f"Stale snapshot after reset for {session_id!r}.")
            current = states.get(session_id)
            actual = current.version if current else 0
            if actual != expected_version:
                raise StateVersionConflict(
                    f"State version conflict for {session_id!r}: expected {expected_version}, found {actual}."
                )
            saved = CognitiveState.from_dict(state.to_dict())
            saved.version, saved.updated_at = expected_version + 1, utc_now()
            saved.created_at = current.created_at if current else state.created_at
            self._publish({**states, session_id: saved}, epoch, resets)
            return CognitiveState.from_dict(saved.to_dict())

    def reset(self, session_id: str | None = None) -> None:
        with self._lock:
            states, epoch, resets = self._committed()
            if session_id is None:
                self._publish({}, epoch + 1, {})
            else:
                sessions = dict(states)
                sessions.pop(session_id, None)
                self._publish(sessions, epoch, {**resets, session_id: resets.get(session_id, 0) + 1})

    def _committed(self) -> tuple[dict[str, CognitiveState], int, dict[str, int]]:
        return self._states, self._epoch, self._resets

    def _publish(self, states: dict[str, CognitiveState], epoch: int, resets: dict[str, int]) -> None:
        self._states, self._epoch, self._resets = states, epoch, resets


class JsonStateManager(InMemoryStateManager):
    """Configured persistent state manager backed by versioned JSON."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._states, self._epoch, self._resets = self._committed()

    def _committed(self) -> tuple[dict[str, CognitiveState], int, dict[str, int]]:
        if not self.path.exists():
            return {}, 0, {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise ValueError("Unsupported cognitive state schema version.")
        states = {key: CognitiveState.from_dict(value) for key, value in payload["sessions"].items()}
        return states, payload.get("epoch", 0), payload.get("resets", {})

    def _publish(self, states: dict[str, CognitiveState], epoch: int, resets: dict[str, int]) -> None:
        payload = {
            "schema_version": "1.0",
            "sessions": {key: value.to_dict() for key, value in states.items()},
            "epoch": epoch,
            "resets": resets,
        }
        temporary = NamedTemporaryFile(dir=self.path.parent, mode="w", encoding="utf-8", delete=False)
        path = Path(temporary.name)
        try:
            with temporary:
                json.dump(payload, temporary, sort_keys=True, allow_nan=False)
            path.replace(self.path)
        finally:
            path.unlink(missing_ok=True)
        super()._publish(states, epoch, resets)
