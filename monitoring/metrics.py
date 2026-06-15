"""Metrics helpers with a safe no-op fallback when OpenTelemetry isn't installed."""
from typing import Any

try:
    from opentelemetry import metrics as _otel_metrics
    _METER = _otel_metrics.get_meter(__name__)
except Exception:
    _METER = None


class _NoOpMetric:
    def add(self, *args, **kwargs):
        return None


def create_counter(name: str, description: str = "") -> Any:
    if _METER is None:
        return _NoOpMetric()
    try:
        return _METER.create_counter(name, description=description)
    except Exception:
        return _NoOpMetric()
