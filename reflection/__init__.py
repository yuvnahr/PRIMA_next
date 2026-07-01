"""PRIMA-NEXT Adaptive Reflection Pillar."""

from typing import Any

__all__ = [
    "AdaptiveReflectionPipeline",
    "FailureType",
    "ReflectionEngine",
    "ReflectionSignal",
    "ReflectionSignalType",
]


def __getattr__(name: str) -> Any:
    if name == "AdaptiveReflectionPipeline":
        from reflection.adaptive_reflection_pipeline import AdaptiveReflectionPipeline

        return AdaptiveReflectionPipeline
    if name == "ReflectionEngine":
        from reflection.reflection_engine import ReflectionEngine

        return ReflectionEngine
    if name == "ReflectionSignal":
        from reflection.reflection_signal import ReflectionSignal

        return ReflectionSignal
    if name in {"FailureType", "ReflectionSignalType"}:
        from reflection.reflection_types import FailureType, ReflectionSignalType

        return {"FailureType": FailureType, "ReflectionSignalType": ReflectionSignalType}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
