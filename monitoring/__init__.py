"""Observability helpers (OpenTelemetry wrappers).

This package contains thin wrappers around OpenTelemetry so the rest of the
application can import helpers without depending directly on OpenTelemetry.
"""

from .health_checks import check_health
from .metrics import create_counter
from .telemetry import init_telemetry
from .tracing import get_tracer, init_tracing

__all__ = ["create_counter", "init_tracing", "get_tracer", "check_health", "init_telemetry"]
