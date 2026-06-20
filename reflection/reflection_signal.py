"""Structured reflection signal."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from reflection.reflection_types import ReflectionSignalType


@dataclass(frozen=True, slots=True)
class ReflectionSignal:
    signal_id: str
    signal_type: ReflectionSignalType
    severity: float
    confidence: float
    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        signal_type: ReflectionSignalType,
        severity: float,
        confidence: float,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> ReflectionSignal:
        return cls(
            signal_id=f"refl_sig_{uuid.uuid4()}",
            signal_type=signal_type,
            severity=max(0.0, min(1.0, severity)),
            confidence=max(0.0, min(1.0, confidence)),
            source=source,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type.value,
            "severity": self.severity,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }
