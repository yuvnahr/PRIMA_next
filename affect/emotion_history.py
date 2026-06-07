"""Rolling emotion history and temporal computations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from affect.emotion_profile import EmotionProfile
from affect.pad_model import PADState


@dataclass(frozen=True, slots=True)
class EmotionHistoryEntry:
    timestamp: datetime
    profile: EmotionProfile
    pad_state: PADState


class EmotionHistory:
    def __init__(self, maxlen: int = 50) -> None:
        self._entries: deque[EmotionHistoryEntry] = deque(maxlen=maxlen)

    def add(
        self,
        profile: EmotionProfile,
        pad_state: PADState,
        timestamp: datetime | None = None,
    ) -> EmotionHistoryEntry:
        entry = EmotionHistoryEntry(timestamp or datetime.now(timezone.utc), profile, pad_state)
        self._entries.append(entry)
        return entry

    def get_recent(self, count: int = 5) -> list[EmotionHistoryEntry]:
        if count <= 0:
            return []
        return list(self._entries)[-count:]

    def compute_velocity(self) -> float:
        recent = self.get_recent(2)
        if len(recent) < 2:
            return 0.0
        return round(recent[-1].pad_state.distance(recent[-2].pad_state), 6)

    def compute_acceleration(self) -> float:
        recent = self.get_recent(3)
        if len(recent) < 3:
            return 0.0
        previous_velocity = recent[-2].pad_state.distance(recent[-3].pad_state)
        current_velocity = recent[-1].pad_state.distance(recent[-2].pad_state)
        return round(current_velocity - previous_velocity, 6)

    def compute_drift(self, baseline: PADState | None = None) -> float:
        if not self._entries:
            return 0.0
        anchor = baseline or self._entries[0].pad_state
        return round(self._entries[-1].pad_state.distance(anchor), 6)

    def average_pad(self, entries: Iterable[EmotionHistoryEntry] | None = None) -> PADState:
        selected = list(entries if entries is not None else self._entries)
        if not selected:
            return PADState()
        return PADState(
            sum(entry.pad_state.pleasure for entry in selected) / len(selected),
            sum(entry.pad_state.arousal for entry in selected) / len(selected),
            sum(entry.pad_state.dominance for entry in selected) / len(selected),
        )

    def __len__(self) -> int:
        return len(self._entries)
